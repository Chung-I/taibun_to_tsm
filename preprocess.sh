mkdir data
python3 preprocess.py raw_data data --prefixes 對齊母語語句 對齊母語字詞
python3 subset_data.py data/all.txt data/dev.txt data/train.txt 1000
python3 subset_data.py data/dev.txt data/test.txt data/dev-tmp.txt 500
mv data/dev-tmp.txt data/dev.txt
