import numpy as np
import math
import torch
import torch.nn as nn

from model import LSTMSeq2seq
from bleu import sentence_bleu
from utils import batch_iter

class BaselineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1 = 128, hidden_size_2 = 64):
        super(BaselineNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size_2, 1)
    
    def forward(self, observations):
        linear_out_1 = self.activation1(self.linear1(observations))
        linear_out_2 = self.activation2(self.linear2(linear_out_1))
        return self.linear3(linear_out_2).squeeze()

class Network(nn.Module):
    def __init__(self, input_size, output_size, action_size):
        super(Network, self).__init__()
        self.rnn = nn.GRU(input_size, output_size, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(output_size * 2, action_size)
        self.softmax = nn.Softmax(dim = 1)
        self.output_size = output_size
    
    def forward(self, observations, observation_lens):
        print("observations size", observations.size())
        print("observations length size", observation_lens.size())
        packed_observation_vectors = torch.nn.utils.rnn.pack_padded_sequence(observations, observation_lens, batch_first=True)
        packed_observation_states, final_states = self.rnn(packed_observation_vectors)
        # _, observation_states = torch.nn.utils.rnn.pad_packed_sequence(packed_observation_states, batch_first=True)
        # print(final_states.size())
        final_states = final_states.view(len(observation_lens), self.output_size * 2)
        actions = self.softmax(self.linear(final_states))
        return actions

class PolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()
        self.lossFunc = nn.CrossEntropyLoss(reduce = False)
    
    def forward(self, pred, target, rewards):
        rewards = (rewards - torch.mean(rewards, 1, True)) / torch.sqrt(torch.var(rewards, 1, True) + 1e-8)
        return torch.sum(rewards * torch.log(self.lossFunc(pred, target)))

class PolicyGradient(nn.Module):
    def __init__(self, vocab, hidden_size, max_decoding_step, pretrain_model_path, \
            average_proportion_factor = -0.4, consecutive_wait_factor = -0.6, average_proportion_baseline = 2, consecutive_wait_baseline = 2):
        super(PolicyGradient, self).__init__()
        self.action_size = 2
        self.vocab = vocab
        self.model = LSTMSeq2seq.load(pretrain_model_path)
        self.input_size = self.model.embedding_size + 2 * self.model.hidden_size
        self.network = Network(self.input_size, hidden_size, self.action_size)
        self.baseline_network = BaselineNetwork(self.input_size)
        self.network_loss = PolicyGradientLoss()
        self.baseline_network_loss = nn.MSELoss()
        self.max_decoding_step = max_decoding_step
        self.state_buffer = []
        self.average_proportion_factor = average_proportion_factor
        self.consecutive_wait_factor = consecutive_wait_factor
        self.average_proportion_baseline = average_proportion_baseline
        self.consecutive_wait_baseline = consecutive_wait_baseline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.IntTensor = torch.cuda.IntTensor
            torch.ByteTensor = torch.cuda.ByteTensor
            torch.FloatTensor = torch.cuda.FloatTensor
            torch.LongTensor = torch.cuda.LongTensor

    def compute_full_BLEU(self, golden, predictions):
        size = golden.size(0)
        result = []
        for i in range(size):
            references = [golden[i]]
            prediction = predictions[i]
            result.append(sentence_bleu(references, prediction)[1])
        
        return torch.FloatTensor(result)

    def compute_partial_BLEU(self, golden, predictions):
        size = golden.size(0)
        result = []
        for i in range(size):
            references = [golden[i]]
            prediction = predictions[i]
            result.append(sentence_bleu(references, prediction)[0])
            
        return torch.FloatTensor(result)

    def forward(self, src, original_src_lens, tgt, otiginal_tgt_lens):
        self.model.eval()
        batch_size = len(src)
        batch_lens = torch.LongTensor(batch_size).fill_(1) # curr READ index
        decoding_steps = torch.LongTensor(batch_size).fill_(0) # curr WRITE index
        ending_step = torch.IntTensor(batch_size).fill_(0)
        ending_flag = torch.ByteTensor(batch_size).fill_(0)
        #TODO: set <s> token in output_tokens
        output_tokens = torch.LongTensor(batch_size, self.max_decoding_step).fill_(0) # B x 1. all <s> as start

        # variable related to rewards
        max_size = self.max_decoding_step + torch.max(original_src_lens)
        rewards = torch.FloatTensor(batch_size, max_size).fill_(0.)
        baseline_rewards = torch.FloatTensor(batch_size, max_size).fill_(0.)

        # variables related to average proportions
        average_proportions = torch.IntTensor(batch_size).fill_(0)
        last_write = torch.IntTensor(batch_size).fill_(0)
        actions = []
        pred_actions = []

        # variables related to consecutive wait length
        consecutive_waits = torch.IntTensor(batch_size).fill_(0)
        
        t = torch.IntTensor(batch_size).fill_(1)
        while torch.max(decoding_steps) < self.max_decoding_step and torch.any(ending_flag != 1):
            sorted_batch_lens, idx_sort = torch.sort(batch_lens, descending=True)
            print(sorted_batch_lens)
            _, idx_unsort = torch.sort(idx_sort, dim=0)
            previous_tokens = torch.gather(output_tokens, 1, decoding_steps.view(-1, 1)).squeeze()
            previous_tokens = torch.index_select(previous_tokens, 0, idx_sort)
            src = torch.index_select(src, 0, idx_sort)
            context_vector, h, prd_token = self.model.forward(src, sorted_batch_lens, previous_tokens)
            prd_embedding = self.model.trg_embedding(prd_token)
            print(context_vector.size(), h.size(), prd_embedding.size())
            # context_vector: B x batch_lens x hidden_size
            # h: B x batch_lens x hidden_size
            # prd_embedding: B x batch_lens x embedding_size
            
            # steps: 
            # 1. compute actions
            # 2. For READ, increment batch_lens
            #    For WRITE, increment decoding_steps
            if len(context_vector.size()) < 3:
                context_vector, h, prd_embedding = context_vector.unsqueeze(1), h.unsqueeze(1), prd_embedding.unsqueeze(1)
            print(context_vector.size(), h.size(), prd_embedding.size())
            observation = torch.cat([context_vector, h, prd_embedding], dim = 2)
            observation = torch.index_select(observation, 0, idx_unsort)
            print(observation.size())
            # t = batch_lens + decoding_steps
            pred_action = self.network(observation, t)
            print(pred_action.size())
            pred_actions.append(pred_action)
            action = torch.ByteTensor(batch_size).fill_(0)

            # READ operations: 
            # 1. increment batch_lens
            read_actions = torch.ByteTensor(batch_size).fill_(0)
            read_actions[(pred_action[:,0] >= 0.5).nonzero()] = 1
            write_actions = ~read_actions
            read_indices = (read_actions & (~ending_flag)).nonzero().view(-1)  # only read(1) and not ending(0) will perform actual write
            action[read_indices] = 1
            batch_lens[read_indices] += 1
            # 2.update 
            consecutive_waits[read_indices] += 1
            # 3. set ending flag for reading the next token of </s>
            read_ending_indices = (batch_lens == original_src_lens).nonzero().view(-1)
            ending_flag[read_ending_indices] = 1
            
            # WRITE operations:
            # 1. increment decoding steps, write to output buffer
            write_indices = (write_actions & (~ending_flag)).nonzero().view(-1) # only write(1) and not ending(0) will perform actual write
            action[write_indices] = 1
            actions.append(action)
            decoding_steps[write_indices] += 1
            # print(write_indices.size())
            # print(decoding_steps[write_indices].size())
            # print(output_tokens[write_indices, decoding_steps[write_indices]].size())
            # print(prd_token[write_indices].size())
            output_tokens[write_indices, decoding_steps[write_indices]] = prd_token[write_indices]
            # 2. update average proportions
            average_proportions[write_indices] += t[write_indices] - last_write[write_indices]
            last_write[write_indices] = t[0]
            
            # 3. update consecutive wait length
            consecutive_waits[write_indices] = 0
            
            # 4. set ending flags for predicting </s>
            # TODO: assume </s> == 1
            write_ending_indices = (prd_embedding == 1).nonzero().view(-1).view(-1)
            ending_flag[write_ending_indices] = 1

            # compute rewards
            ending_indices = torch.cat([read_ending_indices, write_ending_indices], dim = -1).view(-1)
            # print(ending_indices.size())
            print(ending_indices)
            if ending_indices.size(0) > 0:
                ending_step[ending_indices] = t[0]
                rewards[ending_indices, t[0]-1] -= \
                    self.average_proportion_factor * \
                    (average_proportions[ending_indices].to(self.device, dtype=torch.float32) / \
                    (original_src_lens[ending_indices] * decoding_steps[ending_indices]).to(self.device, dtype=torch.float32) \
                        - self.average_proportion_baseline)
                
                # compute BLEU score
                # skip read_indices
                rewards[ending_indices, t[0]-1] += self.compute_full_BLEU(tgt[ending_indices], output_tokens[ending_indices])
            
            non_ending_sequences = ((~ending_flag)).nonzero().view(-1)
            if non_ending_sequences.size(0) > 0:
                rewards[non_ending_sequences, t[0]-1] -= self.consecutive_wait_factor * \
                    ((consecutive_waits[non_ending_sequences] > self.consecutive_wait_baseline) + 1).to(self.device, dtype=torch.float32)

            # compute BLEU score
            write_no_ending_indices = ((write_actions > 0.5) & (~ending_flag)).nonzero().view(-1)
            if write_no_ending_indices.size(0) > 0:
                rewards[write_no_ending_indices, t[0]-1] += self.compute_partial_BLEU(tgt[write_no_ending_indices], output_tokens[write_no_ending_indices])

            # compute baseline rewards
            baseline_reward = self.baseline_network(observation[:,-1,:])
            baseline_rewards[:, t[0]-1] = baseline_reward

            # update time step
            t += 1

        # compute accumulative rewards
        for i in range(1, max_size):
            rewards[:, i] += rewards[:, i - 1]

        # apply mask
        for i in range(batch_size):
            end_indice = ending_step[i]
            rewards[i, ending_indices:] = 0
            baseline_rewards[i, ending_indices:] = 0

        baseline_loss = self.baseline_network_loss(rewards, baseline_rewards)
        new_rewards = rewards - baseline_rewards
        network_loss = self.network_loss(torch.cat(pred_actions), torch.cat(actions), new_rewards)

        return baseline_loss, network_loss, torch.sum(new_rewards)
    
    def validation(self, dev_data, batch_size, cuda=True):
        if torch.cuda.is_available():
            torch.IntTensor = torch.cuda.IntTensor
            torch.ByteTensor = torch.cuda.ByteTensor
            torch.FloatTensor = torch.cuda.FloatTensor
        
        total_network_loss = total_baseline_loss = total_rewards = 0
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            trg_lens = torch.LongTensor(list(map(len, tgt_sents)))

            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            baseline_loss, network_loss, rewards = model(src_sents, src_lens, tgt_sents, trg_lens)
            total_network_loss += network_loss
            total_baseline_loss += baseline_loss
            total_rewards += rewards
        
        print("validation baseline loss %.2f, network loss %.2f, rewards %.2f" %
            (total_baseline_loss / len(dev_data),
            total_network_loss / len(dev_data),
            total_rewards / len(dev_data)), file=sys.stderr)

        return 
