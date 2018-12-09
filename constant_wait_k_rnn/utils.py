import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, orthogonal_
import sentencepiece as spm
import os

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

def lstm_init_(lstm_unit, bidirectional=False):
    '''
    LSTM initialization:

    1) initialize all biases to 0 except forget gate biases
    (since PyTorch has duplicate biases at every LSTM, each forget gate bias
    is initialized to 1/2 instead).

    2) initialized the hidden2hidden matrix by orthogonal

    3) initialized the input2hidden matrix by xavier_uniform
    '''
    for l in range(lstm_unit.num_layers):
        xavier_uniform_(getattr(lstm_unit, "weight_ih_l{}".format(l)).data)
        orthogonal_(getattr(lstm_unit, "weight_hh_l{}".format(l)).data)
        getattr(lstm_unit, "bias_ih_l{}".format(l)).data.fill_(0)
        getattr(lstm_unit, "bias_ih_l{}".format(
            l)).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
        getattr(lstm_unit, "bias_hh_l{}".format(l)).data.fill_(0)
        getattr(lstm_unit, "bias_hh_l{}".format(
            l)).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2

        if bidirectional:
            xavier_uniform_(
                getattr(lstm_unit, "weight_ih_l{}{}".format(l, '_reverse')).data)
            orthogonal_(
                getattr(lstm_unit, "weight_hh_l{}{}".format(l, '_reverse')).data)
            getattr(lstm_unit, "bias_ih_l{}{}".format(l, '_reverse')).data.fill_(0)
            getattr(lstm_unit, "bias_ih_l{}{}".format(l, '_reverse')
                    ).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2
            getattr(lstm_unit, "bias_hh_l{}{}".format(l, '_reverse')).data.fill_(0)
            getattr(lstm_unit, "bias_hh_l{}{}".format(l, '_reverse')
                    ).data[lstm_unit.hidden_size: 2*lstm_unit.hidden_size] = 1./2

def lstm_cell_init_(lstm_cell):
    '''
    Initialize the LSTMCell parameters in a slightly better way
    '''
    xavier_uniform_(lstm_cell.weight_ih.data)
    orthogonal_(lstm_cell.weight_hh.data)
    lstm_cell.bias_ih.data.fill_(0)
    lstm_cell.bias_hh.data.fill_(0)
    lstm_cell.bias_ih.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2
    lstm_cell.bias_hh.data[lstm_cell.hidden_size:2*lstm_cell.hidden_size] = 1./2

class LabelSmoothedCrossEntropy(nn.Module):
    '''
    Args:
        - smoothing_coeff: the smoothing coefficient between target dist and uniform

    Input:
        - pred: (N, C, *)
        - target: (N, * )
    '''
    def __init__(self, smoothing_coeff):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.smoothing_coeff = smoothing_coeff
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        crossent_with_target = self.ce(pred, target)
        crossent_with_uniform = - F.log_softmax(pred).sum(1) / pred.size(1)
        loss = crossent_with_target * self.smoothing_coeff + crossent_with_uniform * (1 - self.smoothing_coeff)
        return loss

def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

# def process_data(file_dir, file_name):
#     data = ""
#     for line in open(file_dir + file_name).readlines():
#         if line.find("<") == -1:
#             doc_text = True
#         else:
#             doc_text = False
#         if doc_text:
#             data += ' '.join(line)[2:]
#
#     with open(file_dir + 'p_' + file_name, 'w') as f:
#         f.write(data)

def process_data(file_dir, file_name):
    data = ""
    for line in open(file_dir + file_name).readlines():
        data += ' '.join(line)

    with open(file_dir + 'p_' + file_name, 'w') as f:
        f.write(data)

# process_data("data/JESC/", "train.ja")
# process_data("data/JESC/", "val.ja")
# process_data("data/JESC/", "test.ja")

def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
