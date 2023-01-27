# List of commands

## Prepare Data Format

CUDA_VISIBLE_DEVICES=0 discopy-extract args-essay /data/discourse/ArgumentAnnotatedEssays-2.0.zip -l 500 --use-gpu |
bzip2 > /cache/discourse/essay.v1.json.bz2

## Train Signal Extraction

## Train Sense Classifier

CUDA_VISIBLE_DEVICES=1 python3 train_conn_sense.py /cache/discourse/pdtb3.en.v3.json.bz2
/cache/discourse/pdtb3.en.v3.roberta.joblib
