# Information Retrieval

## Requirements:
- Python 3
- Java 11
- PIP

## Steps to reproduce
Clone with `--recurse-submodules` to get the required tools folder which is used for evaluation and file conversion.

### Install dependencies
First, run `pip install -r requirements.txt` then afterwards run the following commands:

`pip install --user -U nlt`

`python python3 -m nltk.downloader all`

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
- You may need to change `python` command in the `.sh` and `README` files to `python3` if the `python` command isn't found (depending on your configruation the same goes for `pip` -> `pip3`)
