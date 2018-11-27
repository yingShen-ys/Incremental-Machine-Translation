import torch
import torch.nn.functional as F

from model import LSTMSeq2seq

START_TOKEN_IDX = 1
END_TOKEN_IDX = 2

class MTEBackend(object):
    def __init__(self, class_model_path):
        self.model = LSTMSeq2seq.load(class_model_path, use_gpu=False)
        self.model.training = False
        self.wait_k = 3
        self.src_vocab = self.model.vocab.src
        self.tgt_vocab = self.model.vocab.tgt
        self.reset_state()
    
    def reset_state(self):
        self.src_inputs = []
        self.tgt_outputs = []
        self.curr_decoder_state = None
        self.curr_attn_vector = None

    def predict(self, input_word, inputEnd=False):
        '''
            Given one token, return predicted tokens.
            TODO: Support decoding with one/two tokens.
        '''
        if input_word == '':
            input_word = END_TOKEN_IDX

        input_word_idx = self.src_vocab[input_word]
        self.src_inputs.append(input_word_idx)

        result = []
        decode_token = self.decode_once()
        if decode_token is None and not inputEnd:
            return [None]

        self.tgt_outputs.append(decode_token)
        result.append(self.tgt_vocab.id2word[decode_token])
        if inputEnd:
            while True:
                decode_token = self.decode_once()
                self.tgt_outputs.append(decode_token)
                result.append(self.tgt_vocab.id2word[decode_token])

                if decode_token == END_TOKEN_IDX:
                    break

            self.reset_state()    
        
        return result

    def decode_once(self):
        if len(self.src_inputs) < 3:
            return None

        src_sent = torch.LongTensor(self.src_inputs).unsqueeze(0)
        src_lens = torch.LongTensor([len(self.src_inputs)])

        src_states, final_state = self.model.encode(src_sent, src_lens)
        prev_pred_token = self.tgt_outputs[-1] if len(self.tgt_outputs) > 0 else START_TOKEN_IDX

        prev_pred_token = src_sent.new_ones((1,)).long() * prev_pred_token  # (batch_size,)
        vector = self.model.trg_embedding(prev_pred_token)  # (batch_size, embedding_size)
        
        if self.curr_attn_vector is None:
            vector = torch.cat((vector, vector.new_zeros(vector.size(0), self.model.state_size // self.model.num_layers)),
                            dim=-1)  # input feeding at first step: no previous attentional vector
            h, c = self.model.decoder_lstm_cell(vector, final_state)
            context_vector = self.model.dropout(LSTMSeq2seq.compute_attention(h, src_states, src_lens,
                                                                        attn_func=self.model.attn_func))  # (batch_size, hidden_size (*2))
        else:
            vector = torch.cat((vector, self.curr_attn_vector), dim=-1)  # input feeding
            h, c = self.model.decoder_lstm_cell(vector, self.curr_decoder_state)
            context_vector = self.model.dropout(
                LSTMSeq2seq.compute_attention(h, src_states, src_lens, attn_func=self.model.attn_func))
        
        self.curr_decoder_state = (h, c)

        self.curr_attn_vector = self.model.dropout(
            self.model.decoder_hidden_layer(torch.cat((h, context_vector), dim=-1)))  # the thing to feed in input feeding
        curr_logits = self.model.decoder_output_layer(self.curr_attn_vector)  # (batch_size, vocab_size)
        curr_ll = F.log_softmax(curr_logits, dim=-1)  # transform logits into log-likelihoods
        curr_score, prd_token = torch.max(curr_ll, dim=-1)  # (batch_size,) the decoded tokens
        self.tgt_outputs.append(prd_token.item())
        
        return prd_token.item()

if __name__ == '__main__':
    mtBackend = MTEBackend('work_dir/model.bin')
    input_word = input()
    while(input_word):
        word = mtBackend.predict(input_word)
        if word[0] is None:
            print('please enter next input word')
        else:
            print(word)
        input_word = input()
    
    words = mtBackend.predict(input_word, inputEnd = True)
    print(words)