#!/bin/bash

mkdir -p runs
mkdir -p ranklib-features
mkdir -p models

# Calculate features for train and test test, and write in LETOR format to ranklib-features/
echo "Generating features"
python generate-l2r-features.py

echo "Training model with RankLib"
# ranker 6: LambdaMART
java -jar RankLib.jar -train ranklib-features/data_ranklib-train.txt -ranker 6 -save models/model.txt

echo "Re-rank test set"
java -jar RankLib.jar -load models/model.txt -rank ranklib-features/data_ranklib-test.txt -indri runs/ranklib-score.trec

echo "Evaluating"
tools/eval/trec_eval.9.0.4/trec_eval -c -mmap -mrecip_rank 2019qrels-pass.txt runs/ranklib-score.trec
