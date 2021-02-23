#!/bin/bash

# Create `runs` folder in case is doesn't yet exist
mkdir -p runs

# Perform test queries on indexed data set
python test.py

# Convert test set output from TREC format to the format required by MS-MARCO official evaluation script
python tools/scripts/msmarco/convert_trec_to_msmarco_run.py --input runs/run.msmarco-test2019-queries-bm25.trec --output runs/run.msmarco-test2019-queries-bm25.txt

# Evaluate results from test run with official MS-MARCO evaluation script
python tools/scripts/msmarco/msmarco_passage_eval.py 2019qrels-pass.txt runs/run.msmarco-test2019-queries-bm25.txt

# Extract and compile trec_eval tool if it doesn't exist yet
if [ ! -f "tools/eval/trec_eval.9.0.4/trec_eval" ]; then
  pushd tools/eval
  tar xvfz trec_eval.9.0.4.tar.gz
  cd trec_eval.9.0.4
  make
  popd
fi

# Evaluate results from test run with TREC tool
tools/eval/trec_eval.9.0.4/trec_eval -c -mrecall.1000 -mmap 2019qrels-pass.txt runs/run.msmarco-test2019-queries-bm25.trec