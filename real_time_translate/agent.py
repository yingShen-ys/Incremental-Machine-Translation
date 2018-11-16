import numpy as np
import math
import torch
import torch.nn as nn
import sys
import pickle

from pdb import set_trace
from tqdm import tqdm
from model import LSTMSeq2seq
from bleu import sentence_bleu, corpus_bleu, SmoothingFunction
from utils import batch_iter
from utils import read_corpus, batch_iter, lstm_init_, lstm_cell_init_

def compute_corpus_level_bleu_score(references, hypotheses):
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score

def pad(idx):
    UNK_IDX = 0 # this is built-in into the vocab.py
    max_len = max(map(len, idx))
    for sent in idx:
        sent += [0] * (max_len - len(sent))
    return idx  

def beam_search(model, test_data_src, beam_size, max_decoding_time_step, vocab, cuda):
    was_training = model.training

    model.to('cuda')
    hypotheses = []
    print("BEAM SEARCH IS REPLACED BY GREEDY SEARCH")
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        src_sent = [src_sent] # other parts of the code treat this as a list of sentences...
        src_len = torch.LongTensor(list(map(len, src_sent))).to('cuda')
        src_sent = pad(vocab.src.words2indices(src_sent))
        src_sent = torch.LongTensor(src_sent).to('cuda')
        example_hyps = model.greedy_search(src_sent, src_lens=src_len, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, cuda=cuda)

        hypotheses.append(example_hyps)

    return hypotheses


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

    def forward(self, observation):
        state = self.rnn(observation, self.prev_state)
        action_logits = self.linear(state)
        self.prev_state = state
        return action_logits
    
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
                 discount_factor=1.0):
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
        self.pretrain_model_path = pretrain_model_path
        self.last_rewards = []
        self.last_actions = []
        self.last_action_probs = []
        self.last_decoding_tokens = []
        self.action_softmax = nn.Softmax(dim=1)
        self.smoothing_function = SmoothingFunction().method1

        for param in self.model.parameters():
            param.requires_grad = False

    def load_pretrain_model_weight(self, pretrain_model_path):
        print("Entered loading")
        pretrain_model = LSTMSeq2seq.load(pretrain_model_path)
        self.model = LSTMSeq2seq(pretrain_model.embedding_size, pretrain_model.hidden_size, self.vocab, pretrain_model.dropout.p)
        # self.model.load_state_dict(pretrain_model.state_dict())
        # self.model = pretrain_model.to('cpu')
        own_state = self.model.state_dict()
        for name, param in pretrain_model.state_dict().items():
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

    def compute_full_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction, smoothing_function = self.smoothing_function)[1]

    def compute_partial_BLEU(self, golden, prediction):
        return sentence_bleu(golden, prediction, smoothing_function = self.smoothing_function)[0]

    def test_pretrained(self, args):
        test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
        if args['TEST_TARGET_FILE']:
            test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

        print(f"use the embedded model", file=sys.stderr)
        # model = LSTMSeq2seq.load(args['MODEL_PATH'])
        model = self.model

        vocab = pickle.load(open(args['VOCAB_PATH'], 'rb'))
        hypotheses = beam_search(model, test_data_src,
                                beam_size=1,
                                max_decoding_time_step=70,
                                vocab=vocab, cuda=False)

        if args['TEST_TARGET_FILE']:
            top_hypotheses = [hyps[0] for hyps in hypotheses]
            bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
            print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

        with open(args['OUTPUT_FILE'], 'w') as f:
            for src_sent, hyps in zip(test_data_src, hypotheses):
                top_hyp = hyps[0]
                hyp_sent = ' '.join(top_hyp.value)
                f.write(hyp_sent + '\n')
        return bleu_score

    def check_equal(self, m1, m2):
        for name, param in m1.state_dict().items():
            if name not in m2.state_dict():
                print("Missing parameter in second model")

            if not torch.all(param.data.eq(m2.state_dict()[name].data)):
                print("Parameter {} is different in two models".format(name))

    def forward(self, src, tgt, isTest=False):
        self.model.eval()
        self.network.reset_state()
        tgt = tgt.data.numpy().tolist()
        tgt[0] = tgt[0][1:-1]
        batch_lens = 1 # curr READ index
        decoding_steps = 0 # curr WRITE index
        is_ending = False
        last_context_vector = None
        last_decoder_state = None
        last_bleu_score = 0
        output_tokens = [1]
        observations = []

        # variable related to rewards
        rewards = []

        # variables related to average proportions
        average_proportions = 0
        last_write = 0
        actions = []
        action_logits = []

        # placeholders for some intermediate values for better debugging
        consec_wait_rewards = []
        delta_bleu_rewards = []

        # variables related to consecutive wait length
        consecutive_waits = 0
        t = 0
        while not is_ending:
            rewards.append(torch.zeros(1))
            context_vector, decoder_state, prd_token = self.model(src, torch.LongTensor([batch_lens]), \
                                                                torch.LongTensor([output_tokens[-1]]), last_context_vector, last_decoder_state)

            prd_embedding = self.model.trg_embedding(prd_token)
            last_context_vector = context_vector
            # context_vector: B x batch_lens x hidden_size
            # h: B x batch_lens x hidden_size
            # prd_embedding: B x batch_lens x embedding_size
            # steps: 
            # 1. compute actions
            observation = torch.cat([context_vector, decoder_state[0], prd_embedding], dim = 1)
            # observations.append(observation)
            action_logit = self.network(observation)
            # action_logits.append(action_logit)

            if torch.any(torch.isnan(action_logit)):
                set_trace()
            
            action_prob = self.action_softmax(action_logit)
            m = torch.distributions.Categorical(action_prob)
            action = m.sample()
            # actions.append(action.item())

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
                    # is_ending = True
                    batch_lens = len(src[0])
                    rewards.pop()
                    continue
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
                if len(output_tokens) > 1:
                    full_bleu = self.compute_full_BLEU(tgt, output_tokens[1:])
                    
                    print("output_tokens idx:", output_tokens, file=sys.stderr)
                    print("tgt idx:", tgt, file=sys.stderr)
                    print("output_tokens:", [self.vocab.tgt.id2word.get(output_tokens[i], 'OOV---') for i in range(len(output_tokens[1:]))], file=sys.stderr)
                    print("tgt:", [self.vocab.tgt.id2word.get(tgt[0][i], 'OOV---') for i in range(len(tgt[0]))], file=sys.stderr)
                    print("full bleu:", full_bleu, file=sys.stderr)
                    print("action sequences:", actions, file=sys.stderr)
                    print("", file=sys.stderr)
                else:
                    full_bleu = 0
            else:
                # compute BLEU score
                if len(output_tokens) > 1:
                    bleu_score = self.compute_partial_BLEU(tgt, output_tokens[1:])
                else:
                    bleu_score = 0
                delta_bleu = (bleu_score - last_bleu_score)
                rewards[t] += delta_bleu
                delta_bleu_rewards.append(delta_bleu)
                last_bleu_score = bleu_score

            # sgn(cw - self.cw) + 1 = (2 * int(cw > self.cw) - 1) + 1
            observations.append(observation)
            action_logits.append(action_logit)
            actions.append(action.item())

            consec_wait_reward = self.consecutive_wait_factor * int(consecutive_waits > self.consecutive_wait_baseline) * 2
            rewards[t] += consec_wait_reward
            consec_wait_rewards.append(consec_wait_reward)

            # update time step
            t += 1

            self.last_rewards = rewards
            self.last_actions = actions
            self.last_action_probs = action_logits
            self.last_decoding_tokens = output_tokens

        # compute accumulative rewards
        for i in range(1, len(rewards)):
            rewards[i] *= self.discount_factor ** i # discount
            rewards[i] += rewards[i - 1] + full_bleu + avg_prop_reward # accumulate and add global rewards
        seq_len = len(rewards)
        
        for i in range(1, len(consec_wait_rewards)):
            consec_wait_rewards[i] *= self.discount_factor ** i # discount
        
        for i in range(1, len(delta_bleu_rewards)):
            delta_bleu_rewards[i] *= self.discount_factor ** i # discount

        rewards = torch.cat(rewards)
        baseline_rewards = self.baseline_network(torch.cat(observations, dim=0))
        baseline_loss = self.baseline_network_loss(baseline_rewards, rewards)
        new_rewards = rewards - baseline_rewards # .detach() # detach policy gradients from baseline
        network_loss = self.network_loss(torch.cat(action_logits), torch.tensor(actions), new_rewards)

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
            baseline_loss, network_loss, rewards, _, _, _, _, _, output_tokens, actions = self.forward(src_sents_tensor, tgt_sents_tensor, isTest=True)
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
