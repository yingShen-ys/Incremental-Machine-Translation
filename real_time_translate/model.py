import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_
from collections import namedtuple
# from torch.autograd import Variable
from utils import batch_iter, LabelSmoothedCrossEntropy, lstm_cell_init_, lstm_init_
import time

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
START_TOKEN_IDX = 1
END_TOKEN_IDX = 2

def pad(idx):
    UNK_IDX = 0  # this is built-in into the vocab.py
    max_len = max(map(len, idx))
    for sent in idx:
        sent += [0] * (max_len - len(sent))
    return idx

def dot_attn(a, b):  # computes (batch_size, hidden_size) X (batch_size, max_seq_len, hidden_size) >> (batch_size, max_seq_len, 1)
    return torch.einsum('bi,bji->bj', (a, b)).unsqueeze(-1)

class LSTMSeq2seq(nn.Module):
    '''
    An LSTM based seq2seq model with language as input.

    Args:
         - vocab: the vocab file from vocab.py
         - embedding_size: the size of the embedding
         - hidden_size: the size of the LSTM states, applies to encoder
         - rdrop: whether there is a recurrent dropout on the encoder side
    '''

    def __init__(self, embedding_size, hidden_size, vocab, dropout_rate=0.3, num_layers=2):
        super(LSTMSeq2seq, self).__init__()
        self.state_size = hidden_size * num_layers
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
        # self.softmax = nn.Softmax(dim=1)

    def set_forward_function(self, isPretrain):
        if isPretrain:
            self.forward = self.forward_pretrain
        else:
            self.forward = self.forward_rl

    def forward_pretrain(self, src_tokens, src_lens, trg_tokens, trg_lens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        ll = self.decode_pretrain(src_states, final_states, src_lens, trg_tokens, trg_lens)
        return ll
        
    def forward_rl(self, src_tokens, src_lens, trg_tokens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        context_vector, h, prd_token_embedding = self.decode(src_states, final_states, src_lens, trg_tokens)
        return context_vector, h, prd_token_embedding

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

    def decode_pretrain(self, src_states, final_states, src_lens, trg_tokens, trg_lens):
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
        vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size // self.num_layers)), dim=-1)  # input feeding at first step: no previous attentional vector
        h, c = self.decoder_lstm_cell(vector, final_states)
        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                    attn_func=self.attn_func))  # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
        curr_logits = nn.Softmax(dim=1)(curr_logits)
        _, prd_token = torch.max(curr_logits, dim=-1)  # (batch_size,) the decoded tokens
        
        return context_vector, h, prd_token

    def greedy_search(self, src_sent, src_lens, beam_size=5, max_decoding_time_step=70, cuda=True):
        '''
        Performs beam search decoding for testing the model. Currently just a fake method and only uses argmax decoding.
        '''
        self.training = False  # turn of training
        decoded_idx = []
        scores = 0

        src_states, final_state = self.encode(src_sent, src_lens)
        start_token = src_sent.new_ones((1,)).long() * START_TOKEN_IDX  # (batch_size,) should be </s>
        vector = self.trg_embedding(start_token)  # (batch_size, embedding_size)
        vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size // self.num_layers)),
                           dim=-1)  # input feeding at first step: no previous attentional vector
        h, c = self.decoder_lstm_cell(vector, final_state)
        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                    attn_func=self.attn_func))  # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(
            self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
        curr_ll = F.log_softmax(curr_logits, dim=-1)  # transform logits into log-likelihoods
        curr_score, prd_token = torch.max(curr_ll, dim=-1)  # (batch_size,) the decoded tokens
        decoded_idx.append(prd_token.item())
        scores += curr_score.item()

        decoding_step = 1
        while decoding_step <= max_decoding_time_step and prd_token.item() != END_TOKEN_IDX:
            decoding_step += 1
            vector = self.trg_embedding(prd_token)
            vector = torch.cat((vector, curr_attn_vector), dim=-1)  # input feeding
            h, c = self.decoder_lstm_cell(vector, (h, c))
            context_vector = self.dropout(
                LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=self.attn_func))
            curr_attn_vector = self.dropout(
                self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
            curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
            curr_ll = F.log_softmax(curr_logits, dim=-1)  # transform logits into log-likelihoods
            curr_score, prd_token = torch.max(curr_ll, dim=-1)
            decoded_idx.append(prd_token.item())
            scores += curr_score.item()

        sentence = list(map(lambda x: self.vocab.tgt.id2word[x], decoded_idx))
        if prd_token.item() == END_TOKEN_IDX:
            sentence = sentence[:-1]  # remove the </s> token in final output
        greedy_hyp = Hypothesis(sentence, scores)
        self.training = True  # turn training back on
        return [greedy_hyp] * beam_size
    
    @staticmethod
    def compute_attention(curr_state, src_states, src_lens, attn_func, value_func=None):
        '''
        Computes the context vector from attention.

        Args:
             - curr_state: the current decoder state
             - src_states: the source states of encoder states
             - src_lens: the lengths of the source sequences
             - attn_func: a callback function that computes unnormalized attention scores
                          attn_scores = attn_func(curr_state, src_states)
             - value_func: a function that projects the src_states into another vector space
        '''
        batch_size = curr_state.size(0)
        attn_scores = attn_func(curr_state, src_states)  # (batch_size, max_seq_len)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, src_states.size(1), out=curr_state.new(1).long()).unsqueeze(0)
        mask = (idx < src_lens.unsqueeze(
            1)).float()  # make use of the automatic expansion in comparison. </s> token should receive 0 score
        # mask[:, 0] = 0 # <s> tokens should receive 0 score

        # manual softmax with masking
        offset, _ = torch.max(attn_scores, dim=1, keepdim=True)  # (batch_size, 1, attn_size)
        exp_scores = torch.exp(attn_scores - offset)  # numerical stability (batch_size, max_seq_len, attn_size)
        mask = mask.unsqueeze(-1).expand_as(exp_scores)
        exp_scores = exp_scores * mask
        attn_weights = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)  # (batch_size, max_seq_len, attn_size)

        if value_func is None:
            value_vectors = src_states
        else:
            value_vectors = value_func(src_states)
        context_vector = torch.einsum('bij,bik->bjk', (value_vectors, attn_weights)).sum(-1)
    
        return context_vector
    
    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        model = torch.load(path)
        return model