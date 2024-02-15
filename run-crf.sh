#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1

MODEL_PATH=models/crf_pred

mkdir -p $MODEL_PATH

for i in {1..3}
do
    test_random=$[RANDOM]
    for signal in explicit altlex
    do
        python3 train_conn_crf.py     pdtb3 ${signal} -b 16 --save-path $MODEL_PATH/model_$i --random-seed $test_random
        python3 train_single_sense.py pdtb3 ${signal} -b 16 --save-path $MODEL_PATH/model_$i --random-seed $test_random --simple-sense
    done
    python3 train_conn_sense.py pdtb3 -b 16 --save-path $MODEL_PATH/model_$i --hidden 256,64 --random-seed $test_random --simple-sense
done
