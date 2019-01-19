mkdir data
cd data

mkdir wt103 wt103/models wt103/tmp

cd wt103
wget http://files.fast.ai/models/wt103/*
mv itos_wt103.pkl ./tmp
mv ./* ./models
