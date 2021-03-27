import random
from pyserini.index import IndexReader
from l2r_utils import compute_features, read_topics, format_qrel_line


def get_negative_docid(document_count, docid):
    negative_docid = random.randint(1, document_count)
    while negative_docid == int(docid):
        negative_docid = random.randint(1, document_count)
    return negative_docid


def main(queries_file, qrels_file, output_file, write_negative):
    queries = read_topics(queries_file)
    index_reader = IndexReader('indexes/msmarco-passage')
    document_count = int(index_reader.stats()['documents'])
    qrels = open(qrels_file, 'r')

    with open(output_file, 'w') as output_file_handle:
        for line in qrels:
            line = line.strip().split('\t')

            qid = int(line[0])
            docid = line[2]
            target = line[3]
            query = queries[qid]['title']

            features = compute_features(index_reader, query, docid)
            output_file_handle.write(format_qrel_line(target, qid, features, docid))

            # The evaluation set doesn't need negative examples.
            if write_negative:
                negative_docid = str(get_negative_docid(document_count, docid))
                features = compute_features(index_reader, query, negative_docid)
                output_file_handle.write(format_qrel_line(0, qid, features, negative_docid))


if __name__ == '__main__':
    # Uncommented by default to avoid running a long command (~20 minutes or so)
    # main('queries.train.tsv', 'qrels.train.tsv', 'ranklib-features/data_ranklib-train.txt', True)
    main('msmarco-test2019-queries.tsv', 'runs/run.msmarco-test2019-queries-bm25.trec', 'ranklib-features/data_ranklib-test.txt', False)
