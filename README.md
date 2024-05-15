# List of commands

## Prepare Data Format

CUDA_VISIBLE_DEVICES=0 discopy-extract args-essay /data/discourse/ArgumentAnnotatedEssays-2.0.zip -l 500 --use-gpu |
bzip2 > /cache/discourse/essay.v1.json.bz2

## Train Signal Extraction

python3 train_conn_dis.py pdtb3 -b 256 --save-path models/conn_dis/model_1 --hidden 1024,128 --random-seed 4702
--drop-rate 0.4 -r

## Train Sense Classifier

CUDA_VISIBLE_DEVICES=1 python3 train_conn_sense.py /cache/discourse/pdtb3.en.v3.json.bz2
/cache/discourse/pdtb3.en.v3.roberta.joblib
