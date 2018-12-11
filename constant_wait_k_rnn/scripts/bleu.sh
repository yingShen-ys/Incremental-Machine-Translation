#!/bin/sh
data_dir=$1
k=$2
#test_k=$3
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
    perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode_${test_k}.txt
done
