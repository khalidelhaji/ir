import sys

from nltk.tokenize import word_tokenize
import string
import numpy as np
import time

def get_millisec():
    return int(round(time.time() * 1000))

def load_vectors(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0

    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        if i % 10000 == 0:
            print(i)
        if i == 300000: # 300k
            break
        i += 1

    return data

def load_text(fname):
    fin = open(fname, 'r')
    data = {}

    i = 0
    for line in fin:
        tokens = line.strip().split('\t')
        data[tokens[0]] = tokens[1]

        if i % 10000 == 0:
            print(i)
        i += 1

    return data

def clean(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each wor
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in words if not w in stop_words]

    return words

def compute_mean(vectors, text):
    cleaned = clean(text)

    word_vectors = []
    for word in cleaned:
        if word in vectors:
            word_vectors.append(vectors[word])

    array = np.array(word_vectors)
    mean = np.mean(array, axis=0)
    return np.array2string(mean, max_line_width=sys.maxsize, separator=' ', precision=4, floatmode='fixed', suppress_small=True).strip('[]')

def process_file(vectors, qrels_file, queries_file, docs_file, query_output_file, doc_output_file):
    qrels = open(qrels_file, 'r')
    count = 0

    print('Loading queries')
    queries = load_text(queries_file)
    print('Loading documents')
    documents = load_text(docs_file)

    processed_query_ids = []
    processed_doc_ids = []

    print('Calculating centroids')
    with open(query_output_file, 'w') as query_output_handle:
        with open(doc_output_file, 'w') as doc_output_handle:
            # Write file info according to fastText
            query_output_handle.write('0, 300\n')
            doc_output_handle.write('0, 300\n')

            for qrel in qrels:
                qrel = qrel.strip().split('\t')
                qid = qrel[0]
                docid = qrel[2]

                if qid not in processed_query_ids:
                    query_text = queries[qid]
                    query_mean = compute_mean(vectors, query_text)
                    query_output_handle.write(f'{qid} {query_mean}\n')
                    processed_query_ids.append(qid)

                if docid not in processed_doc_ids:
                    doc_text = documents[docid]
                    doc_mean = compute_mean(vectors, doc_text)
                    doc_output_handle.write(f'{docid} {doc_mean}\n')
                    processed_doc_ids.append(docid)

                if count % 10000 == 0:
                    print(count)
                count += 1

def main():
    print('Loading vectors')
    vectors = load_vectors('crawl-300d-2M.vec')
    # print('Calculating training set')
    # process_file(vectors, 'qrels.train.tsv', 'queries.train.tsv', 'collection.tsv', 'embeddings/queries-embeddings.train.tsv', 'embeddings/documents-embeddings.train.tsv')
    # print('Calculating test set')
    # process_file(vectors, 'runs/run.msmarco-test2019-queries-bm25.trec', 'msmarco-test2019-queries.tsv', 'collection.tsv', 'embeddings/queries-embeddings.test.tsv', 'embeddings/documents-embeddings.test.tsv')


if __name__ == '__main__':
    main()