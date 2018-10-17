import numpy as np
import math
import torch
import torch.nn as nn

from model import Model
from bleu import sentence_bleu
from utils import batch_iter

class BaselineNetwork(nn.modules):
    def __init__(self, input_size, hidden_size_1 = 128, hidden_size_2 = 64):
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size_2, 1)
    
    def forward(self, observations):
        linear_out_1 = self.activation1(self.linear1(observations))
        linear_out_2 = self.activation2(self.linear1(linear_out_1))
        return self.linear1(linear_out_2)

class Network(nn.modules):
    def __init__(self, input_size, output_size, action_size):
        self.rnn = nn.GRU(input_size, output_size, batch_first = True)
        self.linear = nn.Linear(output_size, action_size)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, observations, observation_lens):
        packed_observation_vectors = torch.nn.utils.rnn.pack_padded_sequence(observations, observation_lens, batch_first=True)
        packed_observation_states, final_states = self.rnn(packed_observation_vectors)
        observation_states, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_observations_states, batch_first=True)
        return observation_states

class PolicyGradientLoss(nn.modules):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()
        self.lossFunc = nn.CrossEntropyLoss(reduce = False)
    
    def forward(self, pred, target, rewards):
        rewards = (rewards - torch.mean(rewards, 1, True)) / torch.sqrt(torch.var(rewards, 1, True) + 1e-8)
        return torch.sum(rewards * torch.log(self.lossFunc(pred, target)))

class Reinforce(nn.modules):
    def __init__(self, vocab, hidden_size, max_decoding_step, pretrain_model_path, \
            average_proportion_factor, consecutive_wait_factor, average_proportion_baseline, consecutive_wait_baseline):
        self.action_size = 2
        self.vocab = vocab
        if pretrain_model_path:
            self.model = Model.load(pretrain_model_path)
        else:
            self.model = Model(256, 256, vocab)
        self.input_size = embedding_size + 2 * hidden_size
        self.network = Network(self.input_size, hidden_size, self.action_size)
        self.baseline_network = BaselineNetwork()
        self.network_loss = PolicyGradientLoss()
        self.baseline_network_loss = nn.MSELoss()
        self.max_decoding_step = max_decoding_step
        self.state_buffer = []
        self.average_proportion_factor = average_proportion_factor
        self.consecutive_wait_factor = consecutive_wait_factor
        self.average_proportion_baseline = average_proportion_baseline
        self.consecutive_wait_baseline = consecutive_wait_baseline
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def compute_full_BLEU(self, golden, predictions):
        size = golden.size(0)
        result = []
        for i in range(size):
            references = [golden[i]]
            prediction = predictions[i]
            result.append(sentence_bleu(references, prediction)[1])
        
        return torch.Tensor(result)

    def compute_partial_BLEU(self, golden, predictions):
        size = golden.size(0)
        result = []
        for i in range(size):
            references = [golden[i]]
            prediction = predictions[i]
            result.append(sentence_bleu(references, prediction)[0])
            
        return torch.Tensor(result)

    def train_batch(self, src, original_src_lens, tgt, otiginal_tgt_lens):
        if torch.cuda.is_available():
            torch.IntTensor = torch.cuda.IntTensor
            torch.ByteTensor = torch.cuda.ByteTensor
            torch.FloatTensor = torch.cuda.FloatTensor

        self.model.eval()
        batch_size = len(src)
        batch_lens = torch.IntTensor(batch_size).fill_(1) # curr READ index
        decoding_steps = torch.IntTensor(batch_size).fill_(0) # curr WRITE index
        ending_step = torch.IntTensor(batch_size).fill_(0)
        ending_flag = torch.ByteTensor(batch_size).fill_(0)
        #TODO: set <s> token in output_tokens
        output_tokens = torch.IntTensor([batch_size, self.max_decoding_step]).fill_(0) # B x 1. all <s> as start

        # variable related to rewards
        max_size = self.max_decoding_step + torch.max(original_src_lens)
        rewards = torch.FloatTensor([batch_size, max_size]).fill_(0.)
        baseline_rewards = torch.FloatTensor([batch_size, max_size]).fill_(0.)

        # variables related to average proportions
        average_proportions = torch.IntTensor(batch_size).fill_(0)
        last_write = torch.IntTensor(batch_size).fill_(0)
        actions = []
        pred_actions = []

        # variables related to consecutive wait length
        consecutive_waits = torch.IntTensor(batch_size).fill_(0)
        
        t = torch.Tensor([0 for _ in range(batch_size)], dtype = torch.int32)
        while torch.max(decoding_steps) < self.max_decoding_step and torch.all(ending_flag < 1): #
            context_vector, h, prd_token = self.model.forward(src, batch_lens, output_tokens[..., decoding_steps])
            prd_embedding = self.model.trg_embedding(prd_token)
            # context_vector: B x batch_lens x hidden_size
            # h: B x batch_lens x hidden_size
            # prd_embedding: B x batch_lens x embedding_size
            
            # steps: 
            # 1. compute actions
            # 2. For READ, increment batch_lens
            #    For WRITE, increment decoding_steps
            observation = torch.cat([context_vector, h, prd_embedding], dim = 2)
            # t = batch_lens + decoding_steps
            pred_action = self.network(observation, t)
            pred_actions.append(pred_action)
            action = [0 for _ in range(batch_size)]

            # READ operations: 
            # 1. increment batch_lens
            read_indices = ((pred_action >= 0.5) & (~ending_flag)).nonzero()  # only read(1) and not ending(0) will perform actual write
            action[read_indices] = 1
            batch_lens[read_indices] += 1
            # 2.update 
            consecutive_waits[read_indices] += 1
            # 3. set ending flag for reading the next token of </s>
            ending_indices = (batch_lens == original_src_lens).nonzeros()
            ending_flag[ending_indices] = 1
            
            # WRITE operations:
            # 1. increment decoding steps, write to output buffer
            write_indices = ((pred_action < 0.5) & (~ending_flag)).nonzero() # only write(1) and not ending(0) will perform actual write
            action[write_indices] = 1
            actions.append(action)
            decoding_steps[write_indices] += 1
            output_tokens[decoding_steps[write_indices]] = prd_token[write_indices]
            # 2. update average proportions
            average_proportions[write_indices] += t[write_indices] - last_write[write_indices]
            last_write[write_indices] = t[0]
            
            # 3. update consecutive wait length
            consecutive_waits[write_indices] = 0
            
            # 4. set ending flags for predicting </s>
            # TODO: assume </s> == 1
            ending_indices = (prd_embedding == 1).nonzero()
            ending_flag[ending_indices] = 1

            # compute rewards
            ending_indices = torch.cat([(batch_lens == original_src_lens).nonzeros(), ending_indices = (prd_embedding == 1).nonzero()], dim = 0)
            ending_step[ending_indices] = t[0]
            rewards[ending_indices][t[0]] -= self.average_proportion_factor * \
                (average_proportions[ending_indices]/(original_src_lens[ending_indices] * decoding_steps[ending_indices]) \
                    - self.average_proportion_baseline)
            non_ending_sequences = ((~ending_flag)).nonzero()
            rewards[non_ending_sequences][t[0]] -= self.consecutive_wait_factor * \
                ((consecutive_waits[non_ending_sequences] > self.consecutive_wait_baseline) + 1).to(self.device, dtype=torch.float32)
            
            # compute BLEU score
            # skip read_indices
            rewards[ending_indices] += self.compute_full_BLEU(tgt[ending_indices], output_tokens[ending_indices])
            write_no_ending_indices = ((pred_action > 0.5) & (~ending_flag)).nonzero()
            rewards[write_no_ending_indices] += self.compute_partial_BLEU(tgt[write_no_ending_indices], output_tokens[write_no_ending_indices])

            # compute baseline rewards
            baseline_reward = self.baseline_network(observation)
            baseline_rewards[:, t[0]] = baseline_reward

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
