#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=1
MODEL_PATH=models/ensemble_final

for i in {1..3}
do
    for signal in explicit altlex
    do
        python3 train_conn_labeling.py pdtb3 ${signal} -b 16 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/
        python3 train_single_sense.py pdtb3 ${signal} -b 32 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/
    done
done
