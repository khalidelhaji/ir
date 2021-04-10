#!/bin/bash

# i.e. wiki-news-300d-1M
EMBEDDING_FILE=$1
TRAINING_FEATURES="ranklib-features/$EMBEDDING_FILE/data_ranklib-embeddings-train.txt"
TEST_FEATURES="ranklib-features/$EMBEDDING_FILE/data_ranklib-embeddings-test.txt"
MODEL="models/model-embeddings-$EMBEDDING_FILE.txt"
RUN="runs/ranklib-score-embeddings-$EMBEDDING_FILE.trec"

echo "Training model with RankLib"
# ranker 6: LambdaMART
java -jar RankLib.jar -train $TRAINING_FEATURES -ranker 6 -save $MODEL

echo "Re-rank test set"
java -jar RankLib.jar -load $MODEL -rank $TEST_FEATURES -indri $RUN

echo "Evaluating"
tools/eval/trec_eval.9.0.4/trec_eval -c -mmap -mrecip_rank -mndcg 2019qrels-pass.txt $RUN
