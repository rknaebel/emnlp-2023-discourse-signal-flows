#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=$1

MODEL_PATH=models/crf_altlex_recall

#mkdir -p $MODEL_PATH

#for i in {1..3}
#do
#    test_random=$[RANDOM]
#    python3 train_conn_crf.py   pdtb3 altlex --majority-class-weight 0.01 -b 16 --save-path $MODEL_PATH/model_$i \
#                                             --random-seed $test_random
#    python3 train_conn_sense.py pdtb3 -b 16 --save-path $MODEL_PATH/model_$i --hidden 256,128 \
#                                            --random-seed $test_random --simple-sense --drop-rate 0.4
#done

# acl essays icle nyt pubmed ted unsc
# pdtb3 ted essay bbc unsc anthology nyt aes biodrb biocause
#for corpus in acl essays icle nyt pubmed ted unsc
#do
##    for signal in altlex
##    do
#    signal=altlex
#    python3 predict_signals.py ${corpus} ${signal} "$MODEL_PATH/model_*" -o results/stede-exp/${corpus}.${signal}.csv
##    done
#done

python3 predict_signals.py /cache/discourse/argugpt.v1.json.bz2  explicit "models/crf_pred/model_*" -o results/stede-exp/argugpt.explicit.csv
python3 predict_signals.py /cache/discourse/argugpt.v1.json.bz2  altlex   "$MODEL_PATH/model_*" -o results/stede-exp/argugpt.altlex.csv --use-crf 0
python3 predict_signals.py /cache/stede/anthology.v1.json.bz2    altlex   "$MODEL_PATH/model_*" -o results/stede-exp/acl.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/essay.v3.json.bz2    altlex   "$MODEL_PATH/model_*" -o results/stede-exp/essays.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/icle.v1.json.bz2     altlex   "$MODEL_PATH/model_*" -o results/stede-exp/icle.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/nyt.v5.json.bz2      altlex   "$MODEL_PATH/model_*" -o results/stede-exp/nyt.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/pubmed.v2.json.bz2   altlex   "$MODEL_PATH/model_*" -o results/stede-exp/pubmed.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/ted.v3.json.bz2      altlex   "$MODEL_PATH/model_*" -o results/stede-exp/ted.altlex.csv --use-crf 0
python3 predict_signals.py /cache/discourse/unsc.parla.json.bz2  altlex   "$MODEL_PATH/model_*" -o results/stede-exp/unsc.altlex.csv --use-crf 0