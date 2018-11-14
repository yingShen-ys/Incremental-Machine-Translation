import numpy as np
import math
import torch
import torch.nn as nn
import sys

from pdb import set_trace
from tqdm import tqdm
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
        # self.prev_state = torch.zeros(1, self.output_size).cuda()
        self.prev_state = torch.zeros(1, self.output_size)

        # dirty trick: put a large bias toward READ action so model starts with lot more read
        # self.linear.bias.data[0] += 1e1
    
    def forward(self, observation):
        state = self.rnn(observation, self.prev_state)
        actions = self.softmax(self.linear(state))
        self.prev_state = state
        return actions
    
    def reset_state(self):
        self.prev_state = torch.zeros(1, self.output_size)
        # self.prev_state = torch.zeros(1, self.output_size).cuda()

class PolicyGradientLoss(nn.Module):
    def __init__(self):
        super(PolicyGradientLoss, self).__init__()
        self.lossFunc = nn.CrossEntropyLoss(reduce = False)
    
    def forward(self, pred, target, rewards):
        rewards = (rewards - torch.mean(rewards)) / torch.sqrt(torch.var(rewards) + 1e-8)
        return torch.mean(rewards * self.lossFunc(pred, target))

class PolicyGradient(nn.Module):
    def __init__(self, vocab, hidden_size, max_decoding_step, pretrain_model_path,
                 average_proportion_factor = -0.4, consecutive_wait_factor = -0.6,
                 average_proportion_baseline = 2, consecutive_wait_baseline = 0.5,
                 discount_factor=0.95):
        super(PolicyGradient, self).__init__()
        self.discount_factor = discount_factor
        self.action_size = 2
        self.vocab = vocab
        self.load_pretrain_model_weight(pretrain_model_path)
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

    def load_pretrain_model_weight(self, pretrain_model_path):
        pretrain_model = LSTMSeq2seq.load(pretrain_model_path)
        self.model = LSTMSeq2seq(pretrain_model.embedding_size, pretrain_model.hidden_size, self.vocab, pretrain_model.dropout.p)
        self.model.load_state_dict(pretrain_model.state_dict())
        # own_state = self.model.state_dict()
        # for name, param in pretrain_model.state_dict().items():
        #     if isinstance(param, nn.Parameter):
        #         param = param.data
        #     own_state[name].copy_(param)

    def compute_full_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction)[1]

    def compute_partial_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction)[0]

    def forward(self, src, tgt, isTest=False):
        self.model.eval()
        self.network.reset_state()
        tgt = tgt.data.numpy()
        batch_lens = 1 # curr READ index
        decoding_steps = 0 # curr WRITE index
        is_ending = False
        last_context_vector = None
        last_decoder_state = None
        last_bleu_score = 0
        output_tokens = [0]
        observations = []

        # variable related to rewards
        rewards = []

        # variables related to average proportions
        average_proportions = 0
        last_write = 0
        actions = []
        action_probs = []

        # placeholders for some intermediate values for better debugging
        consec_wait_rewards = []
        delta_bleu_rewards = []

        # variables related to consecutive wait length
        consecutive_waits = 0
        t = 0
        while not is_ending:
            rewards.append(torch.zeros(1))
            # context_vector, h, prd_token = self.model(src, torch.LongTensor([batch_lens]).cuda(), \
            #                                                                         torch.LongTensor([output_tokens[-1]]).cuda(), last_context_vector)
            context_vector, decoder_state, prd_token = self.model(src, torch.LongTensor([batch_lens]), \
                                                torch.LongTensor([output_tokens[-1]]), last_context_vector, last_decoder_state)
            last_context_vector = context_vector
            prd_embedding = self.model.trg_embedding(prd_token)
            # context_vector: B x batch_lens x hidden_size
            # h: B x batch_lens x hidden_size
            # prd_embedding: B x batch_lens x embedding_size
            # steps: 
            # 1. compute actions
            observation = torch.cat([context_vector, decoder_state[0], prd_embedding], dim = 1)
            # observations.append(observation.cpu())
            # action_prob = self.network(observation).cpu()
            observations.append(observation)
            action_prob = self.network(observation)
            action_probs.append(action_prob)
            
            m = torch.distributions.Categorical(action_prob)
            action = m.sample()
            actions.append(action.item())

            # 2. For READ, increment batch_lens
            #    For WRITE, increment decoding_steps
            if action.item() == 0:
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
                last_decoder_state = decoder_state
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
                avg_prop_reward = self.average_proportion_factor * max((average_proportions / (len(src[0]) * len(tgt[0])) - self.average_proportion_baseline), 0) ** 2
                # INSTANT GLOBAL REWARDS SEEM TO NOT BE DISCOUNTED, ADD LATER
                # rewards[t] += avg_prop_reward
                # compute BLEU score
                # skip read_indices
                # print("full score:", self.compute_full_BLEU(tgt, output_tokens))
                full_bleu = self.compute_full_BLEU(tgt, output_tokens)
                # rewards[t] += full_bleu
            else:
                # compute BLEU score
                # print("partial score:", self.compute_partial_BLEU(tgt, output_tokens))
                bleu_score = self.compute_partial_BLEU(tgt, output_tokens)
                delta_bleu = (bleu_score - last_bleu_score)
                rewards[t] += delta_bleu
                delta_bleu_rewards.append(delta_bleu)
                last_bleu_score = bleu_score

            # sgn(cw - self.cw) + 1 = (2 * int(cw > self.cw) - 1) + 1
            consec_wait_reward = self.consecutive_wait_factor * int(consecutive_waits > self.consecutive_wait_baseline) * 2
            rewards[t] += consec_wait_reward
            consec_wait_rewards.append(consec_wait_reward)

            # update time step
            t += 1

        # compute accumulative rewards
        for i in range(1, len(rewards)):
            rewards[i] *= self.discount_factor ** i # discount
            rewards[i] += rewards[i - 1] + full_bleu + avg_prop_reward # accumulate and add global rewards
        
        for i in range(1, len(consec_wait_rewards)):
            consec_wait_rewards[i] *= self.discount_factor ** i # discount
        
        for i in range(1, len(delta_bleu_rewards)):
            delta_bleu_rewards[i] *= self.discount_factor ** i # discount

        rewards = torch.cat(rewards)
        baseline_rewards = self.baseline_network(torch.cat(observations, dim=0))
        baseline_loss = self.baseline_network_loss(baseline_rewards, rewards)
        new_rewards = rewards - baseline_rewards.detach() # detach policy gradients from baseline
        # set_trace()
        network_loss = self.network_loss(torch.cat(action_probs), torch.tensor(actions), new_rewards)

        # set_trace()
        if not isTest:
            return baseline_loss, network_loss, torch.sum(new_rewards), torch.sum(baseline_rewards), avg_prop_reward, sum(consec_wait_rewards), full_bleu, sum(delta_bleu_rewards)

        return baseline_loss, network_loss, torch.sum(new_rewards), torch.sum(baseline_rewards), avg_prop_reward, sum(consec_wait_rewards), full_bleu, sum(delta_bleu_rewards), output_tokens, actions

    def validation(self, dev_data, batch_size=1):
        total_network_loss = total_baseline_loss = total_rewards = 0
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)
            baseline_loss, network_loss, rewards = self.forward(src_sents, tgt_sents)
            total_network_loss += network_loss.item()
            total_baseline_loss += baseline_loss.item()
            total_rewards += rewards

        print("validation baseline loss %.2f, network loss %.2f, rewards %.2f" %
            (total_baseline_loss / len(dev_data),
            total_network_loss / len(dev_data),
            total_rewards / len(dev_data)), file=sys.stderr)
        return

    def test_greedy(self, test_data, test_output, batch_size=1):
        total_network_loss = total_baseline_loss = total_rewards = 0
        f = open(test_output, 'w')
        for src_sents, tgt_sents in tqdm(batch_iter(test_data, batch_size), total=math.ceil(len(test_data) / batch_size)):
            src_sents_tensor = torch.LongTensor(self.vocab.src.words2indices(src_sents))
            tgt_sents_tensor = torch.LongTensor(self.vocab.tgt.words2indices(tgt_sents))
            # set_trace()
            baseline_loss, network_loss, rewards, output_tokens, actions = self.forward(src_sents_tensor, tgt_sents_tensor, isTest=True)
            bleu_score = self.compute_full_BLEU(tgt_sents_tensor.numpy(), output_tokens)
            num_read = num_write = 0
            for i in range(len(output_tokens)):
                output_tokens[i] = self.vocab.tgt.id2word.get(output_tokens[i], 'OOV---')
            for i in range(len(actions)):
                if actions[i] == 0:
                    actions[i] = 'R'
                    num_read += 1
                else:
                    actions[i] = 'W'
                    num_write += 1

            total_network_loss += network_loss.item()
            total_baseline_loss += baseline_loss.item()
            total_rewards += rewards
        
            f.write(' '.join(actions) + '\n')
            f.write(' '.join(output_tokens) + '\n')
            f.write(' '.join(tgt_sents[0]) + '\n')
            f.write(' '.join(src_sents[0]) + '\n')
            f.write("Src length: {}; tgt lenght: {}\n".format(len(src_sents[0]), len(tgt_sents[0])))
            f.write("Read action: {}; write action: {}\n".format(num_read, num_write))
            f.write("BLEU: " + str(bleu_score) + '\n')
            f.write('\n')

        f.close()
        print("validation baseline loss %.2f, network loss %.2f, rewards %.2f" %
            (total_baseline_loss / len(test_data),
            total_network_loss / len(test_data),
            total_rewards / len(test_data)), file=sys.stderr)
