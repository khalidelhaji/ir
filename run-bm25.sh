#!/bin/bash

# Create `runs` folder in case is doesn't yet exist
mkdir -p runs

# Perform test queries on indexed data set
echo "Ranking test queries"
python run-bm25.py

# Convert test set output from TREC format to the format required by MS-MARCO official evaluation script
echo "Converting run"
python tools/scripts/msmarco/convert_trec_to_msmarco_run.py --input runs/run.msmarco-test2019-queries-bm25.trec --output runs/run.msmarco-test2019-queries-bm25.txt

# Filter non relevant judgements
if [ ! -f "2019qrels-pass-filtered.txt" ]; then
  grep -E "[1-9]$" 2019qrels-pass.txt > 2019qrels-pass-filtered.txt
fi

# Evaluate results from test run with official MS-MARCO evaluation script
echo ""
python tools/scripts/msmarco/msmarco_passage_eval.py 2019qrels-pass-filtered.txt runs/run.msmarco-test2019-queries-bm25.txt

# Extract and compile trec_eval tool if it doesn't exist yet
if [ ! -f "tools/eval/trec_eval.9.0.4/trec_eval" ]; then
  pushd tools/eval
  tar xvfz trec_eval.9.0.4.tar.gz
  cd trec_eval.9.0.4
  make
  popd
fi

# Evaluate results from test run with TREC tool
echo "Evaluating"
tools/eval/trec_eval.9.0.4/trec_eval -c -mmap -mrecip_rank -mndcg 2019qrels-pass.txt runs/run.msmarco-test2019-queries-bm25.trec