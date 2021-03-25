# Information Retrieval Core Project

### Requirements
- Python 3.6+
- Java 11
- PIP
- RankLib (make sure the .jar is in this directory)

### Steps to reproduce
Clone with `--recurse-submodules` to get the required tools folder which is used for evaluation and file conversion.

### Install dependencies
First, run `pip install -r requirements.txt` then afterwards run the following commands:

`python -m nltk.downloader all`

### Data
Download from https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019#passage-ranking-dataset
1. Download the 2.9GB MS-MARCO collection `collection.tar.gz`
2. Download `2019qrels-pass.txt`

Put both files in the root directory of this project.

### Commands
1. Run `./generate-index.sh` if you haven't already indexed the 2.9GB MS-MARCO passage collection.
2. Run `./run-bm25.sh` to run and evaluate the test queries with BM25.
3. Run `./run-l2r.sh` to run and evaluate the test queries with L2R.
3. Run `./run-qld.sh` to run and evaluate the test queries with Improvement I.
3. Run `./run-rpf-rm.sh` to run and evaluate the test queries with Improvement II.
3. Run `./run-rpf-rm-mix.sh` to run and evaluate the test queries with Improvement III.
3. Run `./run-rpf-rm3.sh` to run and evaluate the test queries with Improvement IV.

See the `runs` folder for the generated data.

## Troubleshooting
- `chmod +x generate-index.sh run-bm25.sh run-l2r.sh run-qld.sh run-rpf-rm.sh run-rpf-rm-mix.sh run-rpf-rm3.sh` in case GitHub hasn't copied file permission.
- You may need to change `python` command in the `.sh` and `README` files to `python3` if the `python` command isn't found (depending on your configruation the same goes for `pip` -> `pip3`)
