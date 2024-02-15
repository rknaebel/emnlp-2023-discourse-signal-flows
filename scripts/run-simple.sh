#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1
SIGNAL=$2

test_random=$[RANDOM]

MODEL_PATH=models/eval/one-step-weights
python3 train_single_sense.py pdtb3 $SIGNAL -b 32 --save-path $MODEL_PATH --hidden 256,64 --test-set --random-seed $test_random --simple-sense
python3 test_conn_sense.py pdtb3 $SIGNAL --save-path $MODEL_PATH --random-seed $test_random --simple-sense

for w in 1.0 0.8 0.5 0.1 0.05
do
    python3 train_conn_labeling.py pdtb3 ${SIGNAL} -b 32 --save-path $MODEL_PATH/m${w} --test-set --random-seed $test_random --majority-class-weight $w
    cp $MODEL_PATH/best_model_*_sense.pt $MODEL_PATH/m${w}/
    python3 test.py pdtb3 $SIGNAL --save-path $MODEL_PATH/m${w}/ --random-seed $test_random
done
