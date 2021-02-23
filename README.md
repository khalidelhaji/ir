# Information Retrieval

## Steps to reproduce
Clone with `--recurse-submodules` to get the required tools folder which is used for evaluation and file conversion.

### Data
Download from https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#passage-ranking-dataset
1. Download the 2.9GB MS-MARCO collection `collection.tar.gz`
2. Download `2019qrels-pass.txt`

Put both files in the root directory of this project.

### Commands
1. Run `./generate-index.sh` if you haven't already indexed the 2.9GB MS-MARCO passage collection.
2. Run `./run-tests.sh` to run and evaluate the test queries.

## Troubleshooting
- `chmod +x generate-index.sh run-tests.sh` in case GitHub hasn't copied file permission.
