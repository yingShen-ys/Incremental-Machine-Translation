data=$1
python vocab.py --train-src=data/${data}/train.ja --train-tgt=data/${data}/train.en data/${data}/vocab.bin
