#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1

SEEDS=(12193 5449 28708 26410 11508 25960 14813 25735 7315 6550)

for i in {1..10}
do
    test_random=${SEEDS[i-1]}
    for signal in explicit altlex
    do
        python3 test_full.py pdtb3 ${signal} -b 32 --signal-save-path models/eval_conn_label/model_$i/ --sense-save-path models/eval_senses/simple/model_$i/ --random-seed $test_random --simple-sense
        python3 test_full.py pdtb3 ${signal} -b 32 --signal-save-path models/eval_conn_label/model_$i/ --sense-save-path models/eval_senses/std/model_$i/ --random-seed $test_random
    done
done
