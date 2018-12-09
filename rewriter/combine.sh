file=$1
python combine.py --csv /media/bighdd5/ying/tmp/${file}.en.csv --ja /media/bighdd3/ying/tmp/incremental_mt/mt/data/JESC/${file}.ja \
--ja_new /media/bighdd5/ying/tmp/${file}.ja.new --en_new /media/bighdd5/ying/tmp/${file}.en.new
