#!/bin/sh

vocab="data/JESC_tok/vocab.bin"
train_src="data/JESC_tok/train.ja"
train_tgt="data/JESC_tok/train.en"
dev_src="data/JESC_tok/val.ja"
dev_tgt="data/JESC_tok/val.en"
test_src="data/JESC_tok/test.ja"
test_tgt="data/JESC_tok/test.en"

work_dir="work_dir_JESC_tok"

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
    --save-to ${work_dir}/model_512.bin \
    --valid-niter 2400 \
    --batch-size 64 \
    --hidden-size 512 \
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
    ${work_dir}/model_512.bin \
    ${test_src} \
    ${work_dir}/decode_512.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt