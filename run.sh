#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0

for signal in explicit altlex
do
    for corpus in pdtb3 ted essay bbc unsc anthology nyt
    do
        python3 predict_signals.py ${corpus} ${signal} models/ensemble/model_1 -o results/v2/${corpus}.${signal}.csv
    done
done
