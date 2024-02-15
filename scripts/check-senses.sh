#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
MODEL_PATH=models/eval_senses

mkdir -p $MODEL_PATH

for i in {1..10}; do
    test_random=$((RANDOM))

    for signal in explicit altlex; do
        python3 train_single_sense.py pdtb3 ${signal} -b 16 --save-path $MODEL_PATH/simple/model_$i/ --test-set --random-seed $test_random --simple-sense
    done
    python3 train_conn_sense.py pdtb3 -b 16 --save-path $MODEL_PATH/simple/model_$i/ --test-set --random-seed $test_random --simple-sense

    for signal in explicit altlex both; do
        python3 test_conn_sense.py pdtb3 ${signal} -b 32 --save-path $MODEL_PATH/simple/model_$i/ --random-seed $test_random --simple-sense
    done

    for signal in explicit altlex; do
        python3 train_single_sense.py pdtb3 ${signal} -b 16 --save-path $MODEL_PATH/std/model_$i/ --test-set --random-seed $test_random
    done
    python3 train_conn_sense.py pdtb3 -b 16 --save-path $MODEL_PATH/std/model_$i/ --test-set --random-seed $test_random

    for signal in explicit altlex both; do
        python3 test_conn_sense.py pdtb3 ${signal} -b 32 --save-path $MODEL_PATH/std/model_$i/ --random-seed $test_random
    done
done
