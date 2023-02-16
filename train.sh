#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=models/ensemble
DATA_PATH=/cache/discourse/pdtb3.en.v3.json.bz2

for i in {1..3}
do
    for signal in explicit altlex
    do
        python3 train_conn_labeling.py $DATA_PATH ${signal} -b 12 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/ --test-set
        python3 train_single_sense.py pdtb3 ${signal} -b 16 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/ --test-set
    done
    python3 train_conn_sense.py pdtb3 -b 16 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/ --test-set

done
