# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --model-type=<str> --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] VOCAB_PATH MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] VOCAB_PATH MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --ls-rate=<float>                       the smoothing rate of label smoothing [default: 0.9]
    --model-type=<str>                      what type of model to use: lstm, original_lstm, or multi_attn_lstm
    --encoder-layers=<int>                  the number of layers of the encoder model [default: 2]
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time
import torch
import copy
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union, Any
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, lstm_init_, lstm_cell_init_
from vocab import Vocab, VocabEntry
from model import MultiAttnLSTMSeq2seq, OLSTMSeq2seq, LSTMSeq2seq, ScaledAttnLSTMSeq2seq, RecurrentAttnLSTMSeq2seq
from torch.nn.init import uniform_

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
Tensor = torch.tensor

original_print = print

def print_log(content, file=sys.stderr):
    original_print(content)
    original_print(content, file=file)

print = print_log

def pad(idx):
    UNK_IDX = 0 # this is built-in into the vocab.py
    max_len = max(map(len, idx))
    for sent in idx:
        sent += [0] * (max_len - len(sent))
    return idx    

def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
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


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr = float(args['--lr'])

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    if args['--model-type'] == 'lstm':
        pass
    elif args['--model-type'] == 'original_lstm':
        LSTMSeq2seq = OLSTMSeq2seq
    elif args['--model-type'] == 'multi_attn_lstm':
        LSTMSeq2seq = MultiAttnLSTMSeq2seq
    elif args['--model-type'] == 'scaled_attn':
        LSTMSeq2seq = ScaledAttnLSTMSeq2seq
    elif args['--model-type'] == 'rc_attn':
        LSTMSeq2seq = RecurrentAttnLSTMSeq2seq
    else:
        print("Specified model is not implemented!", file=sys.stderr)
        exit(0)

    model = LSTMSeq2seq(embedding_size=int(args['--embed-size']),
                        hidden_size=int(args['--hidden-size']),
                        dropout_rate=float(args['--dropout']),
                        vocab=vocab, label_smooth=float(args['--ls-rate']),
                        num_layers=int(args['--encoder-layers']))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=float(args['--lr-decay']), patience=int(args['--patience']), verbose=True)
    optimizer_state_copy = copy.deepcopy(optimizer.state_dict())

    # uniformly initialize all parameters
    for parameter in model.parameters():
        uniform_(parameter, a=-float(args['--uniform-init']), b=float(args['--uniform-init']))

    # carefully initialize LSTMs
    lstm_init_(model.encoder_lstm)
    lstm_cell_init_(model.decoder_lstm_cell)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    if args['--cuda']:
        torch.LongTensor = torch.cuda.LongTensor
        model.cuda()

    init_tf_rate = 1.0
    tf_rate = init_tf_rate
    decay_steps = len(train_data) * 3
    min_tf_rate = 0.5
    while True:
        epoch += 1
        print(f"There are {len(train_data)//train_batch_size} batches in total.")
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            batch_size = len(src_sents)
            model.train()
            src_lens = torch.LongTensor(list(map(len, src_sents)))
            trg_lens = torch.LongTensor(list(map(len, tgt_sents)))

            src_sents = pad(vocab.src.words2indices(src_sents))
            tgt_sents = pad(vocab.tgt.words2indices(tgt_sents))

            src_sents = torch.LongTensor(src_sents)
            tgt_sents = torch.LongTensor(tgt_sents)

            train_iter += 1
            tf_rate = init_tf_rate - (init_tf_rate - min_tf_rate) * min(train_iter / decay_steps, 1)

            # (batch_size,)
            loss = -model(src_sents, src_lens, tgt_sents, trg_lens, teacher_forcing=tf_rate)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['--clip-grad'])
            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()
            # loss = 0
            report_loss += loss
            cum_loss += loss

            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d / %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter, len(train_data)//train_batch_size,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.eval()
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl
                lr_scheduler.step(valid_metric)

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                # also compute dev set BLEU
                dev_hypotheses = greedy_search(model, dev_data_src,
                                        beam_size=int(args['--beam-size']),
                                        max_decoding_time_step=int(args['--max-decoding-time-step']),
                                        vocab=vocab, cuda=args['--cuda'])

                dev_best_hypotheses = list(map(lambda x: x[0], dev_hypotheses))
                dev_bleu = compute_corpus_level_bleu_score(dev_data_tgt, dev_best_hypotheses)
                print(f"dev set BLEU score: {dev_bleu}", file=sys.stderr)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    torch.save(model, model_save_path)
                    optimizer_state_copy = copy.deepcopy(optimizer.state_dict())

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = LSTMSeq2seq.load(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        optimizer.load_state_dict(optimizer_state_copy)
                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: object, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, vocab: Vocab, cuda: str) -> List[List[Hypothesis]]:
    was_training = model.training

    model.to('cuda')
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        src_sent = [src_sent] # other parts of the code treat this as a list of sentences...
        src_len = torch.LongTensor(list(map(len, src_sent))).to('cuda')
        src_sent = pad(vocab.src.words2indices(src_sent))
        src_sent = torch.LongTensor(src_sent).to('cuda')
        example_hyps = model.beam_search(src_sent, src_lens=src_len, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, cuda=cuda)

        hypotheses.append(example_hyps)

    return hypotheses


def greedy_search(model: object, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int, vocab: Vocab, cuda: str) -> List[List[Hypothesis]]:
    was_training = model.training

    model.to('cuda')
    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        src_sent = [src_sent] # other parts of the code treat this as a list of sentences...
        src_len = torch.LongTensor(list(map(len, src_sent))).to('cuda')
        src_sent = pad(vocab.src.words2indices(src_sent))
        src_sent = torch.LongTensor(src_sent).to('cuda')
        example_hyps = model.greedy_search(src_sent, src_lens=src_len, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step, cuda=cuda)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = LSTMSeq2seq.load(args['MODEL_PATH'])

    vocab = pickle.load(open(args['VOCAB_PATH'], 'rb'))
    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']),
                             vocab=vocab, cuda=args['--cuda'])

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)
    if args['--cuda']:
        torch.cuda.manual_seed_all(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
