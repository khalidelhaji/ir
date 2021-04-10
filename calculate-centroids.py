import sys
import os
import string
import time
import numpy as np
from nltk.tokenize import word_tokenize
from l2r_utils import load_fasttext_vectors


def load_text(fname):
    fin = open(fname, 'r')
    data = {}

    i = 0
    for line in fin:
        tokens = line.strip().split('\t')
        data[tokens[0]] = tokens[1]

        if i % 100000 == 0:
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


def process_file(is_training):
    embeddings_file = 'embeddings/glove.6B.300d'
    docs_file = 'collection.tsv'
    if is_training:
        qrels_file = 'qrels.train.tsv'
        queries_file = 'queries.train.tsv'
        query_output_file = f'{embeddings_file}/queries-embeddings.train.tsv'
        doc_output_file = f'{embeddings_file}/documents-embeddings.train.tsv'
    else:
        qrels_file = 'runs/run.msmarco-test2019-queries-bm25.trec'
        queries_file = 'msmarco-test2019-queries.tsv'
        query_output_file = f'{embeddings_file}/queries-embeddings.test.tsv'
        doc_output_file = f'{embeddings_file}/documents-embeddings.test.tsv'

    print('Loading vectors')
    vectors = load_fasttext_vectors(f'{embeddings_file}.vec')
    print('Loading queries')
    queries = load_text(queries_file)
    print('Loading documents')
    documents = load_text(docs_file)

    qrels = open(qrels_file, 'r')
    count = 0
    processed_query_ids = []
    processed_doc_ids = []

    print('Calculating centroids')
    os.system(f'mkdir -p {embeddings_file}')
    with open(query_output_file, 'w') as query_output_handle:
        with open(doc_output_file, 'w') as doc_output_handle:
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


if __name__ == '__main__':
    start_time = round(time.time())
    print(f'Starting at: {start_time}')
    process_file(False)
    end_time = round(time.time())
    print(f'Done at: {end_time}')
    print(f'Duration: {end_time - start_time}')
