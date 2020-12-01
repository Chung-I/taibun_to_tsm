data_dir=data
mkdir $data_dir
python3 preprocess.py raw_data $data_dir --tat
python3 subset_data.py $data_dir/all.txt $data_dir/dev.txt $data_dir/train.txt 200
python3 subset_data.py $data_dir/dev.txt $data_dir/test.txt $data_dir/dev-tmp.txt 100
mv $data_dir/dev-tmp.txt $data_dir/dev.txt
