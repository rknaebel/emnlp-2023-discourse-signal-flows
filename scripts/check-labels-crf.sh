#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
MODEL_PATH=models/eval_conn_label

mkdir -p $MODEL_PATH

for i in {1..10}
do
    test_random=$[RANDOM]
    for signal in explicit altlex
    do
        python3 train_conn_crf.py pdtb3 ${signal} -b 16 --save-path $MODEL_PATH/model_$i/ --test-set --random-seed $test_random
        python3 test_conn_crf.py pdtb3 ${signal} -b 32 $MODEL_PATH/model_$i/ --random-seed $test_random
    done
done

#+ python3 test_conn_crf.py pdtb3 altlex -b 32 models/test_labeling_crf/model_1/ --random-seed 9074
#+ python3 test_conn_crf.py pdtb3 altlex -b 32 models/test_labeling_crf/model_2/ --random-seed 13570
