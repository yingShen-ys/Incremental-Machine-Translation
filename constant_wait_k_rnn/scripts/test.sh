#!/bin/sh
data_dir=$1
k=$2
test_k=$3
vocab="data/${data_dir}/vocab.bin"
train_src="data/${data_dir}/train.ja"
train_tgt="data/${data_dir}/train.en"
dev_src="data/${data_dir}/val.ja"
dev_tgt="data/${data_dir}/val.en"
test_src="data/${data_dir}/test.ja"
test_tgt="data/${data_dir}/test.en"

work_dir="work_dir_${k}"

for test_k in {1..16}
do
    echo ${test_k}
    python nmt.py \
        decode \
        --beam-size 5 \
        --max-decoding-time-step 100 \
        ${vocab} \
        ${work_dir}/model.bin \
        ${test_src} \
        ${work_dir}/decode_${test_k}.txt \
        --wait-k ${test_k}
done
perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode_${test_k}.txt
