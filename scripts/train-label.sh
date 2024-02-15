#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=models/ensemble_predict
#DATA_PATH=/cache/discourse/pdtb3.en.v3.json.bz2

for i in {1..3}
do
    for signal in explicit altlex
    do
        python3 train_conn_labeling.py pdtb3 ${signal} -b 8 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/
    done
done
d