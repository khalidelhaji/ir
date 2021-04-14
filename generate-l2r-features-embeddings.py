import random
import math
import os
import time
from pyserini.index import IndexReader
from l2r_utils import compute_features, read_topics, format_qrel_line, load_fasttext_vectors, load_fasttext_line
from scipy import spatial


def compute_similarity(query_vector, doc_vector):
    return {
        'cosine_similarity': 1 - spatial.distance.cosine(query_vector, doc_vector)
    }


def get_negative_docid(available_doc_ids, docid):
    negative_docid = random.choice(available_doc_ids)
    while int(negative_docid) == int(docid):
        negative_docid = random.choice(available_doc_ids)
    return negative_docid


def main(is_training):
    embeddings_file = 'glove.840B.300d'
    print(f'Processing {embeddings_file}')

    if is_training:
        qrels_file = 'qrels.train.tsv'
        queries_file = 'queries.train.tsv'
        query_embeddings_file = f'embeddings/{embeddings_file}/queries-embeddings.train.tsv'
        doc_embeddings_file = f'embeddings/{embeddings_file}/documents-embeddings.train.tsv'
        output_file = f'ranklib-features/{embeddings_file}/data_ranklib-embeddings-train.txt'
    else:
        qrels_file = 'runs/run.msmarco-test2019-queries-bm25.trec'
        queries_file = 'msmarco-test2019-queries.tsv'
        query_embeddings_file = f'embeddings/{embeddings_file}/queries-embeddings.test.tsv'
        doc_embeddings_file = f'embeddings/{embeddings_file}/documents-embeddings.test.tsv'
        output_file = f'ranklib-features/{embeddings_file}/data_ranklib-embeddings-test.txt'

    queries = read_topics(queries_file)
    index_reader = IndexReader('indexes/msmarco-passage')
    qrels = open(qrels_file, 'r')

    print('Reading query vectors')
    query_embeddings_handle = open(query_embeddings_file, 'r')
    query_vector_id, query_vector_values = load_fasttext_line(query_embeddings_handle.readline())

    print('Reading document vectors')
    doc_vectors = load_fasttext_vectors(doc_embeddings_file, False)
    doc_ids = list(doc_vectors.keys())

    count = 0
    print('Calculating features')
    os.system(f'mkdir -p ranklib-features/{embeddings_file}')
    with open(output_file, 'w') as output_file_handle:
        for line in qrels:
            line = line.strip().split('\t')

            qid = int(line[0])
            docid = line[2]
            target = line[3]
            query = queries[qid]['title']

            if int(query_vector_id) != qid:
                old_id = query_vector_id
                while int(old_id) == int(query_vector_id):
                    query_vector_id, query_vector_values = load_fasttext_line(query_embeddings_handle.readline())

            doc_vector = doc_vectors[docid]
            if math.isnan(query_vector_values[0]) or math.isnan(doc_vector[0]):
                count += 1
                continue

            features = {
                **compute_similarity(query_vector_values, doc_vector),
                **compute_features(index_reader, query, docid)
            }
            output_file_handle.write(format_qrel_line(target, qid, features, docid))

            # The evaluation set doesn't need negative examples.
            if is_training:
                negative_docid = str(get_negative_docid(doc_ids, docid))
                features = {
                    **compute_similarity(query_vector_values, doc_vectors[negative_docid]),
                    **compute_features(index_reader, query, docid)
                }
                output_file_handle.write(format_qrel_line(0, qid, features, negative_docid))

            if count % 10000 == 0:
                print(count)
            count += 1


if __name__ == '__main__':
    start_time = round(time.time())
    print(f'Starting at: {start_time}')
    main(False)
    end_time = round(time.time())
    print(f'Done at: {end_time}')
    print(f'Duration: {end_time - start_time}')
