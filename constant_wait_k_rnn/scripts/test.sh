#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"


model_dir="/media/bighdd5/zhun/mt/assignment1/work_dir_tfa_drop_if_ls9_big"
work_dir="work_dir"

python nmt_mk_iii.py \
    decode \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${vocab} \
    ${model_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt


perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt