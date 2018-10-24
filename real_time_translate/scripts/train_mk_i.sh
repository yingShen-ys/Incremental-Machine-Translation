#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

mkdir -p ${work_dir}
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
    --cuda \
    --seed 233 \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/model_rl.bin \
    --valid-niter 2400 \
    --batch-size 1 \
    --hidden-rl-size 512 \
    --uniform-init 0.1 \
    --pretrain-model-path ${work_dir}/model.bin \
    --max-decoding-time-step 40 \
    --average-proportion-factor -0.5 \
    --consecutive-wait-factor -0.5 \
    --average-proportion-baseline 2 \
    --consecutive-wait-baseline 2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log

# python main.py \
#     decode \
#     --seed 233 \
#     --beam-size 5 \
#     --max-decoding-time-step 100 \
#     ${vocab} \
#     ${work_dir}/model.bin \
#     ${test_src} \
#     ${work_dir}/decode.txt

# perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt