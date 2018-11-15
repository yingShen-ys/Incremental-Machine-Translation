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

    def set_forward_function(self, isPretrain):
        if isPretrain:
            self.forward = self.forward_pretrain
        else:
            self.forward = self.forward_rl

    def forward_pretrain(self, src_tokens, src_lens, trg_tokens, trg_lens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        ll = self.decode_pretrain(src_states, final_states, src_lens, trg_tokens, trg_lens)
        return ll
        
    def forward_rl(self, src_tokens, src_lens, trg_tokens, last_context_vector, decoder_hidden_state):
        src_states, final_states = self.encode(src_tokens, src_lens)
        context_vector, h, prd_token = self.decode(src_states, final_states, src_lens, trg_tokens, last_context_vector, decoder_hidden_state)
        return context_vector, h, prd_token

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









############################################################################################################################


    def forward_true(self, src_tokens, src_lens, trg_tokens):
        src_states, final_states = self.encode(src_tokens, src_lens)
        prd_tokens = self.decode_true(src_states, final_states, src_lens, trg_tokens, torch.tensor([trg_tokens.size(1)]))
        return prd_tokens

    def decode_true(self, src_states, final_states, src_lens, trg_tokens, trg_lens):
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
        output = []

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
        output.append(prd_token.item())
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
            output.append(prd_token.item())
            prd_token = trg_tokens[..., t + 2]

        # computing the masked log-likelihood
        nll = torch.stack(nll, dim=1)

        # create masks, assuming padding is AT THE END
        idx = torch.arange(0, trg_tokens.size(1), out=trg_tokens.new(1).long()).unsqueeze(0)
        mask = (idx < trg_lens.unsqueeze(1)).float()  # make use of the automatic expansion in comparison
        assert nll.size() == mask[:,
                             1:].size(), f"Negative log-likelihood has shape {nll.size()}, yet the mask has shape {mask[:, 1:].size()}"
        masked_log_likelihoods = - nll * mask[:, 1:]  # exclude <s> token

        return output  # seems the training code assumes the log-likelihoods are summed per word


################################################################################################################################






    def decode(self, src_states, final_states, src_lens, trg_tokens, last_context_vector = None, decoder_hidden_state = None):
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
        if last_context_vector is None:
            # input feeding at first step: no previous attentional vector
            vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size // self.num_layers)), dim=-1)
        else:
            vector = torch.cat((vector, last_context_vector), dim=-1)
        
        if decoder_hidden_state is None:
            h, c = self.decoder_lstm_cell(vector, final_states)
        else:
            h, c = self.decoder_lstm_cell(vector, decoder_hidden_state)
        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                    attn_func=self.attn_func))  # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.decoder_output_layer(curr_attn_vector)  # (batch_size, vocab_size)
        curr_logits = F.log_softmax(curr_logits, dim=-1)
        _, prd_token = torch.max(curr_logits, dim=-1)  # (batch_size,) the decoded tokens
        
        return context_vector, (h, c), prd_token

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

    def beam_search(self, src_sent, src_lens, beam_size=5, max_decoding_time_step=70, cuda=True, feed_embedding=False):
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
            value: List[str]: the decoded target sentence, represented as a list of words
            score: float: the log-likelihood of the target sentence
        """
        self.training = False # turn of training
        if cuda:
            torch.FloatTensor = torch.cuda.FloatTensor

        decoded_beam_idx = []
        bk_pointers = [[-1]]

        batch_size = src_sent.size(0)
        src_states, final_state = self.encode(src_sent, src_lens) # (1, src_lens, hidden)

        # decode start token
        start_token = src_sent.new_ones((1,)).long() * START_TOKEN_IDX # (batch_size,) should be </s>
        vector = self.trg_embedding(start_token) # (batch_size, embedding_size)
        vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.state_size//self.num_layers)), dim=-1) # input feeding at first step
        h, c = self.decoder_lstm_cell(vector, final_state)

        # compute weighted sum of layers hidden states
        # normalized_layer_weights = F.softmax(self.decoder_lstm_layer_unnorm_weights, dim=-1)
        # reshaped_h = h.view(batch_size, self.decoder_layers, -1)
        # h = torch.einsum('k,bkj->bj', (normalized_layer_weights, reshaped_h))

        context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=self.attn_func)) # (batch_size, hidden_size (*2))
        curr_attn_vector = self.dropout(self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1))) # the thing to feed in input feeding
        curr_logprobs = self.decoder_output_layer(curr_attn_vector) # (batch_size, vocab_size)
        # curr_ll = F.log_softmax(curr_logprobs, dim=-1) # transform logits into log-likelihoods
        best_scores, best_score_ids = torch.topk(curr_logprobs, beam_size, dim=-1) # (batch_size, beam_size)
        best_beam_scores = best_scores # (batch_size, beam_size)
        bk_pointer = best_score_ids / self.trg_vocab_size  # (batch_size, beam_size)
        best_score_ids = best_score_ids - bk_pointer * self.trg_vocab_size
        decoded_beam_idx.append(best_score_ids)
        _, prd_token = torch.max(curr_logprobs, dim=-1)

        # expand h, c, src_states, curr_attn_vector for next beam_size tokens: (batch, ) -> (batch * beam_size, )
        h = h.data.repeat(1, beam_size).view(-1, h.size(-1))
        c = c.data.repeat(1, beam_size).view(-1, c.size(-1))
        src_states = src_states.data.repeat(1, beam_size, 1).view(-1, src_states.size(1), src_states.size(2))
        curr_attn_vector = curr_attn_vector.data.repeat(1, beam_size).view(-1, curr_attn_vector.size(-1))

        survived_size = beam_size
        survived_score = best_beam_scores
        survived_pos = None
        survived_id = decoded_beam_idx[-1].view(-1)
        src_states_tmp = src_states
        finished_scores = torch.cuda.FloatTensor([])
        finished_pos = []

        # decode target sentences
        for t in range(1, max_decoding_time_step):
            vectors = self.trg_embedding(survived_id.view(-1, survived_size)) # (batch_size, survived_size) -> (batch_size, survived_size, embedding_size)
            vectors = vectors.view(-1, self.embedding_size) # (batch_size, survived_size, embedding_size) -> (batch_size * survived_size, embedding_size)
            vectors = torch.cat((vectors, curr_attn_vector), dim=-1) # input feeding again...
            h, c = self.decoder_lstm_cell(vectors, (h, c))

            # compute weighted sum of layers hidden states
            # normalized_layer_weights = F.softmax(self.decoder_lstm_layer_unnorm_weights, dim=-1)
            # reshaped_h = h.view(batch_size * survived_size, self.decoder_layers, -1)
            # h = torch.einsum('k,bkj->bj', (normalized_layer_weights, reshaped_h))

            context_vector = self.dropout(LSTMSeq2seq.compute_attention(h, src_states_tmp, src_lens, attn_func=self.attn_func))

            curr_attn_vector = self.dropout(self.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1))) # the thing to feed in input feeding
            curr_logprobs = self.decoder_output_layer(curr_attn_vector) # (batch_size, vocab_size)
            # curr_ll = F.log_softmax(curr_logprobs, dim=-1) # transform logits into log-likelihoods
            scores = (curr_logprobs + survived_score.view(-1, 1)).view(-1, self.trg_vocab_size*survived_size) # (batch_size, survived_size * vocab_size)
            best_scores, best_score_ids = torch.topk(scores, beam_size, dim=-1) # (batch_size, beam_size)
            best_beam_scores = best_scores
            # recalculate bk_pointer and ids
            bk_pointer = best_score_ids / self.trg_vocab_size # (batch_size, beam_size)
            best_score_ids = best_score_ids - bk_pointer * self.trg_vocab_size
            bk_pointer_o = bk_pointer.view(-1)
            if survived_pos is not None: # recalculate bk_pointer
                bk_pointer = survived_pos[bk_pointer]
            # append decoded beam
            bk_pointers.append(bk_pointer)
            decoded_beam_idx.append(best_score_ids)

            # check for </s>
            end_id = best_score_ids.view(-1).data.eq(END_TOKEN_IDX)
            survived_id = best_score_ids.view(-1)[~end_id]
            survived_pos = (~end_id).nonzero().view(-1)
            finished_num = end_id.nonzero().view(-1).size()[0]
            survived_size = beam_size - finished_num
            survived_score = best_beam_scores.view(-1)[~end_id]
            
            if t == max_decoding_time_step-1:
                finished_scores = torch.cat((finished_scores, best_beam_scores.view(-1) / float(t)))
                # finished_scores = torch.cat((finished_scores, best_beam_scores.view(-1)))
                finished_pos.extend([(t, i) for i in range(0, beam_size)])
            elif finished_num > 0: # add finished sentence
                finished_scores = torch.cat((finished_scores, best_beam_scores.view(-1)[end_id] / float(t)))
                # finished_scores = torch.cat((finished_scores, best_beam_scores.view(-1)[end_id]))
                finished_pos.extend([(t, end_id.nonzero().view(-1)[i].item()) for i in range(0, finished_num)])

            if survived_size == 0:
                break

            # prepare h, c based on bk_pointer
            prev_id = bk_pointer_o.view(-1)[survived_pos]
            h = h[prev_id]
            curr_attn_vector = curr_attn_vector[prev_id]
            c = c[prev_id]
            src_states_tmp = src_states[:survived_size, :, :]

            assert survived_id.size()[0] == h.size()[0]

        # sort finished score and finished pos
        best_scores, best_score_ids = torch.topk(torch.FloatTensor(finished_scores.cpu()), beam_size)
        best_score_pos = [finished_pos[i.item()] for i in best_score_ids]

        # back track
        beam_hyps = []
        for b in range(0, beam_size):
            token_pos = best_score_pos[b] # (t, i)
            pos = token_pos[1]
            sentence = []
            for bk_i in range(token_pos[0], 0, -1):
                pos = bk_pointers[bk_i][0, pos].item()
                token = decoded_beam_idx[bk_i - 1][0, pos].item()
                sentence.append(token)

            sentence = list(map(lambda x: self.vocab.tgt.id2word[x], reversed(sentence)))
            beam_hyps.append(Hypothesis(sentence, best_scores[b].item()))
        self.training = True # turn training back on
        return beam_hyps

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
    
    def evaluate_ppl(self, dev_data, batch_size, cuda=True):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """
        self.training = False
        cum_loss = 0.
        cum_trg_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        if cuda:
            torch.LongTensor = torch.cuda.LongTensor

        for src_sents, trg_sents in batch_iter(dev_data, batch_size):
            trg_word_num_to_predict = sum(len(s[1:]) for s in trg_sents)  # omitting the leading `<s>`
            cum_trg_words += trg_word_num_to_predict
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            trg_lens = torch.LongTensor(list(map(len, trg_sents)))

            # these padding functions modify data in-place
            src_sents = pad(self.vocab.src.words2indices(src_sents))
            trg_sents = pad(self.vocab.tgt.words2indices(trg_sents))

            src_sents = torch.LongTensor(src_sents)
            trg_sents = torch.LongTensor(trg_sents)
            loss = -self.forward(src_sents, src_lens, trg_sents, trg_lens).sum()

            loss = loss.item()
            cum_loss += loss

        ppl = np.exp(cum_loss / cum_trg_words)
        self.training = True
        return ppl
    
    def save_model(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location='cpu')
        return model