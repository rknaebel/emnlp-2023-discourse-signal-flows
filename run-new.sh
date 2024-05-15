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

#python3 predict_signals.py /cache/discourse/argugpt.v1.json.bz2  explicit "models/crf_pred/model_*" -o results/stede-exp/argugpt.explicit.csv
#python3 predict_signals.py /cache/discourse/argugpt.v1.json.bz2  altlex   "$MODEL_PATH/model_*" -o results/stede-exp/argugpt.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/stede/anthology.v1.json.bz2    altlex   "$MODEL_PATH/model_*" -o results/stede-exp/acl.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/essay.v3.json.bz2    altlex   "$MODEL_PATH/model_*" -o results/stede-exp/essays.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/icle.v1.json.bz2     altlex   "$MODEL_PATH/model_*" -o results/stede-exp/icle.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/nyt.v5.json.bz2      altlex   "$MODEL_PATH/model_*" -o results/stede-exp/nyt.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/pubmed.v2.json.bz2   altlex   "$MODEL_PATH/model_*" -o results/stede-exp/pubmed.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/ted.v3.json.bz2      altlex   "$MODEL_PATH/model_*" -o results/stede-exp/ted.altlex.csv --use-crf 0
#python3 predict_signals.py /cache/discourse/unsc.parla.json.bz2  altlex   "$MODEL_PATH/model_*" -o results/stede-exp/unsc.altlex.csv --use-crf 0

MODEL_PATH=models/conn_dis

#mkdir -p $MODEL_PATH
#
#for i in {1..3}
#do
#    test_random=$[RANDOM]
#    python3 train_conn_dis.py pdtb3 -b 256 --save-path $MODEL_PATH/model_$i --hidden 1024,128 --random-seed $test_random --drop-rate 0.4 -r
#done

#+ python3 train_conn_dis.py pdtb3 -b 256 --save-path models/conn_dis/model_1 --hidden 1024,128 --random-seed 4702 --drop-rate 0.4 -r
#+ python3 train_conn_dis.py pdtb3 -b 256 --save-path models/conn_dis/model_2 --hidden 1024,128 --random-seed 357 --drop-rate 0.4 -r
#+ python3 train_conn_dis.py pdtb3 -b 256 --save-path models/conn_dis/model_3 --hidden 1024,128 --random-seed 7611 --drop-rate 0.4 -r


# STEDE
#python3 predict_conns.py /cache/discourse/argugpt.v1.json.bz2  "$MODEL_PATH/model_*" -o results/stede-exp/argugpt.conns.csv -r
#python3 predict_conns.py /cache/stede/anthology.v1.json.bz2    "$MODEL_PATH/model_*" -o results/stede-exp/acl.conns.csv -r
#python3 predict_conns.py /cache/discourse/essay.v3.json.bz2    "$MODEL_PATH/model_*" -o results/stede-exp/essays.conns.csv -r
#python3 predict_conns.py /cache/discourse/icle.v1.json.bz2     "$MODEL_PATH/model_*" -o results/stede-exp/icle.conns.csv -r
#python3 predict_conns.py /cache/discourse/ted.v3.json.bz2      "$MODEL_PATH/model_*" -o results/stede-exp/ted.conns.csv -r
#python3 predict_conns.py /cache/discourse/unsc.parla.json.bz2  "$MODEL_PATH/model_*" -o results/stede-exp/unsc.conns.csv -r
#python3 predict_conns.py /cache/discourse/nyt.v5.json.bz2      "$MODEL_PATH/model_*" -o results/stede-exp/nyt.conns.csv -r
#python3 predict_conns.py /cache/discourse/pubmed.v2.json.bz2   "$MODEL_PATH/model_*" -o results/stede-exp/pubmed.conns.csv -r

# RESULTS PAPER CONNS
python3 predict_conns.py anthology  "$MODEL_PATH/model_*" -o results/v2/anthology.conns.csv --limit 10000
python3 predict_conns.py essay      "$MODEL_PATH/model_*" -o results/v2/essays.conns.csv --limit 5000
python3 predict_conns.py aes        "$MODEL_PATH/model_*" -o results/v2/aes.conns.csv --limit 5000
python3 predict_conns.py ted        "$MODEL_PATH/model_*" -o results/v2/ted.conns.csv --limit 5000
python3 predict_conns.py unsc       "$MODEL_PATH/model_*" -o results/v2/unsc.conns.csv --limit 5000
python3 predict_conns.py bbc        "$MODEL_PATH/model_*" -o results/v2/bbc.conns.csv --limit 5000
python3 predict_conns.py pdtb3      "$MODEL_PATH/model_*" -o results/v2/pdtb3.conns.csv --limit 5000
python3 predict_conns.py nyt        "$MODEL_PATH/model_*" -o results/v2/nyt.conns.csv --limit 50000
python3 predict_conns.py pubmed     "$MODEL_PATH/model_*" -o results/v2/pubmed.conns.csv --limit 10000
