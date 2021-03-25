import math
import random
import json
from pyserini.index import IndexReader
import io
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from scipy.spatial.distance import cosine

TOTAL_DOCS = 8841823
LAMBDA = 0.5  # LMIR.JM hyperparameter
MU = 2000  # LMIR.DIR hyperparameter


def read_topics(file_name):
    file = open(file_name)

    topics = {}
    for line in file:
        line = line.strip().split("\t")
        # Try and parse the keys into integers
        try:
            topic_key = int(line[0])
        except ValueError:
            topic_key = line[0]
        topics[topic_key] = {
            'title': line[1],
        }

    return topics

def get_vector_mean(vectors, words):
    word_vectors = []
    for word in words:
        if word in vectors:
            word_vectors.append(vectors[word])
    return pd.DataFrame(word_vectors).mean()

def compute_similarity(vectors, query, doc):
    query_words = clean(query)
    doc_words = clean(doc)

    query_mean = get_vector_mean(vectors, query_words)
    doc_mean = get_vector_mean(vectors, doc_words)

    # list1 = []
    # for word in query_words:
    #     if word in vectors:
    #         list1.append(word)
    # list2 = []
    # for word in doc_words:
    #     if word in vectors:
    #         list2.append(word)
    # print(list1)
    # print(list2)

    return 1 - cosine(query_mean, doc_mean)

def compute_features(index_reader, vectors, query, qid, docid, target):
    # BM25 score
    bm25_score = index_reader.compute_query_document_score(docid, query)

    # TF info
    tf = index_reader.get_document_vector(docid)
    total_terms = sum(tf.values())

    # Cumulative variables
    tf_idf_sum = 0
    jm = 1
    dirich = 1

    for word in tf.keys():
        if word in query:
            counts = index_reader.get_term_counts(word, analyzer=None)
            df = counts[0]  # document frequency
            cf = counts[1]  # collection frequency

            # TF-IDF
            idf = math.log(math.e, (TOTAL_DOCS / max(df, 1)))
            tf_idf_sum += (tf[word] / total_terms) * idf

            # Jelineck-Mercer (JM) smoothing (LMIR.JM)
            p_ml = tf[word] / total_terms
            p_global = tf[word] / max(cf, 1)
            jm *= (1 - LAMBDA) * p_ml + LAMBDA * p_global

            # Dirichlet smoothing (LMIR.DIR), the probabilites exceed 1 here for some reason
            dirich *= (tf[word] + MU * cf) / (total_terms + MU)

    # doc = json.loads(index_reader.doc_raw(docid))["contents"]
    # similarity = compute_similarity(vectors, query, doc)
    similarity = 0

    bm25_score = "{:.8f}".format(bm25_score)
    tf_idf_sum = "{:.8f}".format(tf_idf_sum)
    jm = "{:.8f}".format(jm)
    dirich = "{:.8f}".format(dirich)
    similarity = "{:.8f}".format(similarity)
    line_format = f"{target} qid:{qid} 1:{bm25_score} 2:{tf_idf_sum} 3:{total_terms} 4:{jm} 5:{dirich} 6:{similarity} #{docid}\n"
    # print(line_format)
    return line_format


def get_negative_docid(document_count, docid):
    negative_docid = random.randint(1, document_count)
    while negative_docid == int(docid):
        negative_docid = random.randint(1, document_count)
    return negative_docid


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))

        if i % 10000 == 0:
            print(i)
        if i == 10000: # 300k
            break
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

def main(vectors, queries_file, qrels_file, output_file, separator, write_negative):
    queries = read_topics(queries_file)
    index_reader = IndexReader('indexes/msmarco-passage')
    document_count = int(index_reader.stats()['documents'])
    qrels = open(qrels_file, 'r')

    count = 0

    with open(output_file, 'w') as output_file_handle:
        for line in qrels:
            line = line.strip().split(separator)

            qid = int(line[0])
            docid = line[2]
            target = line[3]
            query = queries[qid]['title']

            output_file_handle.write(compute_features(index_reader, vectors, query, qid, docid, target))

            # The evaluation set doesn't need negative examples.
            if write_negative:
                negative_docid = str(get_negative_docid(document_count, docid))
                output_file_handle.write(compute_features(index_reader, vectors, query, qid, negative_docid, 0))

            if count % 10000 == 0:
                print(count)
            count += 1


if __name__ == '__main__':
    print('Loading vectors')
    vectors = load_vectors('crawl-300d-2M.vec')

    print('Calculating features')
    # Uncommented by default to avoid running a long command (~20 minutes or so)
    main(vectors, 'queries.train.tsv', 'qrels.train.tsv', 'ranklib-features/data_ranklib-train.txt', '\t', True)
    # main(vectors, 'msmarco-test2019-queries.tsv', 'runs/run.msmarco-test2019-queries-bm25.trec', 'ranklib-features/data_ranklib-test.txt', ' ', False)
