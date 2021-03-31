import random
import math
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
    if is_training:
        queries_file = 'queries.train.tsv'
        qrels_file = 'qrels.train.tsv'
        query_embeddings_file = 'embeddings/queries-embeddings.train.tsv'
        doc_embeddings_file = 'embeddings/documents-embeddings.train.tsv'
        output_file = 'ranklib-features/data_ranklib-embeddings-train.txt'
    else:
        queries_file = 'msmarco-test2019-queries.tsv'
        qrels_file = 'runs/run.msmarco-test2019-queries-bm25.trec'
        query_embeddings_file = 'embeddings/queries-embeddings.test.tsv'
        doc_embeddings_file = 'embeddings/documents-embeddings.test.tsv'
        output_file = 'ranklib-features/data_ranklib-embeddings-test.txt'

    queries = read_topics(queries_file)
    index_reader = IndexReader('indexes/msmarco-passage')
    document_count = int(index_reader.stats()['documents'])
    qrels = open(qrels_file, 'r')

    print('Reading query vectors')
    # query_vectors = load_fasttext_vectors(query_embeddings_file, False)
    query_embeddings_handle = open(query_embeddings_file, 'r')
    query_embeddings_handle.readline() # First line only contains some statistics on the file.
    query_vector_id, query_vector_values = load_fasttext_line(query_embeddings_handle.readline())

    print('Reading document vectors')
    doc_vectors = load_fasttext_vectors(doc_embeddings_file, False)
    doc_ids = list(doc_vectors.keys())

    count = 0
    print('Calculating features')
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
            if math.isnan(query_vector_values[0]):
                count += 1
                continue
            # if int(query_vector_id) != qid:
            #     print('Something went wrong')
            #     print(f'qid: {qid}')
            #     print(f'vector_id: {query_vector_id}')
            #     exit(0)

            features = {
                **compute_similarity(query_vector_values, doc_vectors[docid]),
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
    main(True)
