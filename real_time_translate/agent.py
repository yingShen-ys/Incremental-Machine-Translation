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
        self.rnn = nn.GRUCell(input_size, output_size)
        self.linear = nn.Linear(output_size, action_size)
        self.softmax = nn.Softmax(dim = 1)
        self.output_size = output_size
        self.prev_state = torch.zeros(1, self.output_size)
    
    def forward(self, observation):
        state = self.rnn(observation, self.prev_state)
        actions = self.softmax(self.linear(state))
        self.prev_state = state
        return actions
    
    def reset_state(self):
        self.prev_state = torch.zeros(1, self.output_size)

class PolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()
        self.lossFunc = nn.CrossEntropyLoss(reduce = False)
    
    def forward(self, pred, target, rewards):
        rewards = (rewards - torch.mean(rewards)) / torch.sqrt(torch.var(rewards) + 1e-8)
        return torch.mean(rewards * self.lossFunc(pred, target))

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
        for param in self.model.parameters():
            param.requires_grad = False
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        #     torch.IntTensor = torch.cuda.IntTensor
        #     torch.ByteTensor = torch.cuda.ByteTensor
        #     torch.FloatTensor = torch.cuda.FloatTensor
        #     torch.LongTensor = torch.cuda.LongTensor

    def compute_full_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction)[1]

    def compute_partial_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction)[0]

    def forward(self, src, tgt):
        self.model.eval()
        self.network.reset_state()
        tgt = tgt.data.numpy()
        batch_lens = 1 # curr READ index
        decoding_steps = 0 # curr WRITE index
        is_ending = False
        last_context_vector = None
        output_tokens = [0]
        observations = []

        # variable related to rewards
        rewards = []

        # variables related to average proportions
        average_proportions = 0
        last_write = 0
        actions = []
        action_probs = []

        # variables related to consecutive wait length
        consecutive_waits = 0
        t = 0
        while not is_ending:
            rewards.append(torch.zeros(1))
            context_vector, h, prd_token = self.model(src, torch.LongTensor([batch_lens]).cuda(), \
                                                                                    torch.LongTensor([output_tokens[-1]]).cuda(), last_context_vector)
            last_context_vector = context_vector
            prd_embedding = self.model.trg_embedding(prd_token)
            # context_vector: B x batch_lens x hidden_size
            # h: B x batch_lens x hidden_size
            # prd_embedding: B x batch_lens x embedding_size
            
            # steps: 
            # 1. compute actions
            context_vector = context_vector.cpu()
            h = h.cpu()
            prd_embedding = prd_embedding.cpu()
            observation = torch.cat([context_vector, h, prd_embedding], dim = 1)
            observations.append(observation)
            action_prob = self.network(observation)
            action_probs.append(action_prob)
            
            m = torch.distributions.Categorical(action_prob)
            action = m.sample()
            actions.append(action.data[0])

            # 2. For READ, increment batch_lens
            #    For WRITE, increment decoding_steps
            if action.data[0] == 0:
                # READ operations: 
                # 1. increment batch_lens
                batch_lens += 1
                # 2.update 
                consecutive_waits += 1
                # 3. set ending flag for reading the next token of </s>
                if batch_lens == len(src[0]) + 1:
                    is_ending = True
            else:
                # WRITE operations:
                # 1. increment decoding steps, write to output buffer
                decoding_steps += 1
                output_tokens.append(prd_token.item())
                # 2. update average proportions
                average_proportions += t - last_write
                last_write = t
                
                # 3. update consecutive wait length
                consecutive_waits = 0
                
                # 4. set ending flags for predicting </s>
                if prd_token.item() == 2 or len(output_tokens) >= self.max_decoding_step:
                    is_ending = True

            # compute rewards
            if is_ending:
                rewards[t] += self.average_proportion_factor * (average_proportions / (len(src[0]) * len(tgt[0])) - self.average_proportion_baseline)
                # compute BLEU score
                # skip read_indices
                rewards[t] += self.compute_full_BLEU(tgt, output_tokens)
            else:
                # compute BLEU score
                rewards[t] += self.compute_partial_BLEU(tgt, output_tokens)

            rewards[t] += self.consecutive_wait_factor * ((consecutive_waits > self.consecutive_wait_baseline) + 1)

            # update time step
            t += 1

        # compute accumulative rewards
        for i in range(1, len(rewards)):
            rewards[i] += rewards[i - 1]

        rewards = torch.cat(rewards)
        baseline_rewards = self.baseline_network(torch.cat(observations, dim=0))
        baseline_loss = self.baseline_network_loss(baseline_rewards, rewards)
        new_rewards = rewards - baseline_rewards
        network_loss = self.network_loss(torch.cat(action_probs), torch.stack(actions), new_rewards)

        return baseline_loss, network_loss, torch.sum(new_rewards)
    
    def validation(self, dev_data, batch_size, cuda=True):
        total_network_loss = total_baseline_loss = total_rewards = 0
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            baseline_loss, network_loss, rewards = model(src_sents, tgt_sents)
            total_network_loss += network_loss.item(0)
            total_baseline_loss += baseline_loss.item(0)
            total_rewards += rewards
        
        print("validation baseline loss %.2f, network loss %.2f, rewards %.2f" %
            (total_baseline_loss / len(dev_data),
            total_network_loss / len(dev_data),
            total_rewards / len(dev_data)), file=sys.stderr)

        return 
