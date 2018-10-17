import numpy as np
import math
import torch
import torch.nn as nn

class Model(nn.Module):
    '''
    An LSTM based seq2seq model with language as input.

    Args:
         - vocab: the vocab file from vocab.py
         - embedding_size: the size of the embedding
         - hidden_size: the size of the LSTM states, applies to encoder
         - bidirectional: whether or not the encoder is bidirectional
         - rdrop: whether there is a recurrent dropout on the encoder side
    '''

    def __init__(self, embedding_size, hidden_size, vocab, dropout_rate=0.3, num_layers=2):
        super(LSTMSeq2seq, self).__init__()
        self.state_size = (hidden_size * 2 if bidirectional else hidden_size * 1) * num_layers
        self.vocab = vocab
        self.src_vocab_size = len(vocab.src)
        self.trg_vocab_size = len(vocab.tgt)
        self.src_embedding = nn.Embedding(self.src_vocab_size, embedding_size)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, embedding_size)
        self.encoder_lstm = nn.LSTM(embedding_size, hidden_size, dropout=dropout_rate, num_layers = num_layers)
        self.decoder_lstm_cell = nn.LSTMCell(embedding_size + self.state_size // num_layers,
                                             self.state_size // num_layers)
        self.decoder_hidden_layer = nn.Linear(2 * self.state_size // num_layers, self.state_size // num_layers)
        self.decoder_output_layer = nn.Linear(self.state_size // num_layers, self.trg_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_func = dot_attn
        self.enc_final_to_dec_init = nn.Linear(self.state_size, self.state_size // num_layers)
        self.num_layers = num_layers
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward_pretrain(self, src_tokens, src_lens, trg_tokens, trg_lens, teacher_forcing=0.5):
        src_states, final_states = self.encode(src_tokens, src_lens)
        ll = self.decode_pretrain(src_states, final_states, src_lens, trg_tokens, trg_lens, teacher_forcing=teacher_forcing)
        return ll

    def encode(self, src_tokens, src_lens):
        '''
        Encode source sentences into vector representations.

        Args:
             - src_tokens: a torch tensor of a batch of tokens, with shape (batch_size, max_seq_len) >> LongTensor
             - src_lens: a torch tensor of the sentence lengths in the batch, with shape (batch_size,) >> LongTensor
        '''
        src_vectors = self.src_embedding(src_tokens)  # (batch_size, max_seq_len, embedding_size)
        packed_src_vectors = torch.nn.utils.rnn.pack_padded_sequence(src_vectors, src_lens, batch_first=True)
        packed_src_states, final_states = self.encoder_lstm(packed_src_vectors)  # both (batch_size, max_seq_len, hidden_size (*2))

        src_states, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_src_states, batch_first=True)
        final_cell_states = final_states[-1].permute(1, 0, 2)

        # use a linear mapping to bridge encoder and decoder
        batch_size = final_cell_states.size(0)
        c = self.enc_final_to_dec_init(final_cell_states.contiguous().view(batch_size, -1))
        h = torch.tanh(c)
        return src_states, (h, c)

    def decode_pretrain(self, src_states, final_states, src_lens, trg_tokens, trg_lens, teacher_forcing=0.5):
        '''
        Decode with attention and custom decoding.

        Args:
             - src_states: the source sentence encoder states at different time steps
             - final_states: the last state of input source sentences
             - src_lens: the lengths of source sentences, helpful in computing attention
             - trg_tokens: target tokens, used for computing log-likelihood as well as teacher forcing (if toggled True)
             - trg_lens: target sentence lengths, helpful in computing the loss
             - teacher_forcing: whether or not the decoder sees the gold sequence in previous steps when decoding
             - search_method: greedy, beam_search, etc. Not yet implemented.
        '''
        nll = []

        # dealing with the start token
        start_token = trg_tokens[..., 0]  # (batch_size,)
        vector = self.trg_embedding(start_token)  # (batch_size, embedding_size)
        vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size // self.num_layers)),
                           dim=-1)  # input feeding at first step: no previous attentional vector
        h, c = self.decoder_lstm_cell(vector, final_states)
        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                    attn_func=self.attn_func))  # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(
            self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
        neg_log_likelihoods = self.ce_loss(curr_logits, trg_tokens[..., 1])  # (batch_size,)
        nll.append(neg_log_likelihoods)
        _, prd_token = torch.max(curr_logits, dim=-1)  # (batch_size,) the decoded tokens
        if np.random.uniform() < teacher_forcing:
            prd_token = trg_tokens[..., 1]  # feed the gold sequence token to the next time step

        for t in range(trg_tokens.size(-1) - 2):
            token = prd_token  # trg_tokens[:, t+1]
            vector = self.trg_embedding(token)
            vector = torch.cat((vector, curr_attn_vector), dim=-1)  # input feeding
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = self.dropout(
                LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=self.attn_func))
            curr_attn_vector = self.dropout(
                self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
            curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
            neg_log_likelihoods = self.ce_loss(curr_logits, trg_tokens[..., t + 2])  # (batch_size,)
            nll.append(neg_log_likelihoods)
            _, prd_token = torch.max(curr_logits, dim=-1)
            if np.random.uniform() < teacher_forcing:
                prd_token = trg_tokens[..., t + 2]

        # computing the masked log-likelihood
        nll = torch.stack(nll, dim=1)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, trg_tokens.size(1), out=trg_tokens.new(1).long()).unsqueeze(0)
        mask = (idx < trg_lens.unsqueeze(1)).float()  # make use of the automatic expansion in comparison
        assert nll.size() == mask[:,
                             1:].size(), f"Negative log-likelihood has shape {nll.size()}, yet the mask has shape {mask[:, 1:].size()}"
        masked_log_likelihoods = - nll * mask[:, 1:]  # exclude <s> token

        return torch.sum(masked_log_likelihoods)  # seems the training code assumes the log-likelihoods are summed per word
    
    def forward(self, src_tokens, src_lens, trg_tokens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        context_vector, h, prd_token_embedding = self.decode(src_states, final_states, src_lens, trg_tokens)
        return context_vector, h, prd_token_embedding

    def decode(self, src_states, final_states, src_lens, trg_tokens):
        '''
        Decode with attention and custom decoding.

        Args:
             - src_states: the source sentence encoder states at different time steps
             - final_states: the last state of input source sentences
             - src_lens: the lengths of source sentences, helpful in computing attention
             - trg_tokens: target tokens, used for computing log-likelihood as well as teacher forcing (if toggled True)
             - trg_lens: target sentence lengths, helpful in computing the loss
             - teacher_forcing: whether or not the decoder sees the gold sequence in previous steps when decoding
             - search_method: greedy, beam_search, etc. Not yet implemented.
        '''
        start_token = trg_tokens  # (batch_size,)
        vector = self.trg_embedding(start_token)  # (batch_size, embedding_size)
        vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size // self.num_layers)),
                           dim=-1)  # input feeding at first step: no previous attentional vector
        h, c = self.decoder_lstm_cell(vector, final_states)
        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                    attn_func=self.attn_func))  # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
        curr_logits = nn.Softmax(curr_logits)
        _, prd_token = torch.max(curr_logits, dim=-1)  # (batch_size,) the decoded tokens
        
        return context_vector, h, prd_token
    
    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load_model(self, path):
        model = torch.load(path)
        return model