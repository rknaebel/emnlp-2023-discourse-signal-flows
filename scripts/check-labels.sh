#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=models/ensemble_labels_new

mkdir -p $MODEL_PATH

for i in {1..3}
do
    test_random=$[RANDOM]
    for signal in explicit altlex
    do
        python3 train_conn_labeling.py pdtb3 ${signal} -b 16 --split-ratio 0.9 --save-path $MODEL_PATH/model_$i/ --test-set --random-seed $test_random
        python3 test_conn_labeling.py pdtb3 ${signal} -b 32 --save-path $MODEL_PATH/model_$i/ --random-seed $test_random
    done
done

#echo "-- TEST RESULTS"
#for i in {1..3}
#do
#    for signal in explicit altlex
#    do
#        python3 test_conn_labeling.py pdtb3 ${signal} -b 32 --save-path $MODEL_PATH/model_$i/
#    done
#done

#python3 test_conn_labeling.py pdtb3 explicit -b 32 --save-path "$MODEL_PATH/model_*/"
#python3 test_conn_labeling.py pdtb3 altlex -b 32 --save-path "$MODEL_PATH/model_*/"
