#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

# mkdir -p ${work_dir}
echo save results to ${work_dir}

python main.py \
    pretrain \
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

python main.py \
    train \
    --seed 233 \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model_rl2.bin \
    --valid-niter 4800 \
    --batch-size 1 \
    --hidden-rl-size 512 \
    --uniform-init 0.1 \
    --pretrain-model-path ${work_dir}/model.bin \
    --max-decoding-time-step 40 \
    --average-proportion-factor -0.5 \
    --consecutive-wait-factor -0.5 \
    --average-proportion-baseline 0.3 \
    --consecutive-wait-baseline 2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --network-lr=0.00001 \
    --baseline-lr=0.0001 2>${work_dir}/err2.log

python main.py \
    test \
    --seed 233 \
    --max-decoding-time-step 40 \
    ${vocab} \
    ${work_dir}/model_rl2.bin1920 \
    ${test_src} \
    ${test_tgt} \
    ${work_dir}/decode.txt

# perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt