data_dir=$1
k=$2
vocab="data/${data_dir}/vocab.bin"
train_src="data/${data_dir}/train.ja"
train_tgt="data/${data_dir}/train.en"
dev_src="data/${data_dir}/val.ja"
dev_tgt="data/${data_dir}/val.en"
test_src="data/${data_dir}/test.ja"
test_tgt="data/${data_dir}/test.en"

work_dir="work_dir_${k}"


mkdir -p ${work_dir}
echo save results to ${work_dir}

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
    --valid-niter 12646 \
    --batch-size 256 \
    --hidden-size 378 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr 0.001 \
    --lr-decay 0.5 2>${work_dir}/err.log \
    --wait-k ${k}


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
