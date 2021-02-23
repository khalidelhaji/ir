#!/bin/bash

# Clear existing collection files
rm -rf collections/msmarco-passage || true
mkdir -p collections/msmarco-passage

python tools/scripts/msmarco/convert_collection_to_jsonl.py \
  --collection-path collection.tsv \
  --output-folder collections/msmarco-passage

python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
  -threads 9 -input collections/msmarco-passage \
  -index indexes/msmarco-passage -storePositions -storeDocvectors -storeRaw
