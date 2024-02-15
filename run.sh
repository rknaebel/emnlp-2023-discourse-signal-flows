#!/bin/bash
set -x

CUDA_VISIBLE_DEVICES=$1

for corpus in pdtb3 ted essay bbc unsc anthology nyt aes biodrb biocause
do
    for signal in explicit altlex
    do
        python3 predict_signals.py ${corpus} ${signal} "models/crf_pred/model_*" --limit 3000 -o results/v2/${corpus}.${signal}.csv
    done
done

for signal in explicit altlex
do
    python3 predict_signals.py nyt-full ${signal} "models/crf_pred/model_*" -o results/v2/nyt-full.${signal}.csv
done
