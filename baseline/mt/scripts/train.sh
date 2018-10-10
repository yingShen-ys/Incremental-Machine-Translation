#!/bin/sh

vocab="data/JESC/vocab.bin"
train_src="data/JESC/train.ja"
train_tgt="data/JESC/train.en"
dev_src="data/JESC/val.ja"
dev_tgt="data/JESC/val.en"
test_src="data/JESC/test.ja"
test_tgt="data/JESC/test.en"

work_dir="work_dir_JESC"

#mkdir -p ${work_dir}
#echo save results to ${work_dir}

#python vocab.py --train-src=data/JESC/train.ja --train-tgt=


python nmt.py \
    train \
    --model-type original_lstm \
    --cuda \
    --seed 233 \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model.bin \
    --valid-niter 2400 \
    --batch-size 64 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log

python nmt.py \
    decode \
    --seed 233 \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${vocab} \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt