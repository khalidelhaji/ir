import math
import re

TOTAL_DOCS = 8841823
LAMBDA = 0.5  # LMIR.JM hyperparameter
MU = 2000  # LMIR.DIR hyperparameter


def load_fasttext_line(line):
    tokens = list(filter(None, line.rstrip().split(' ')))
    return tokens[0], list(map(float, tokens[1:]))


def load_fasttext_vectors(file_name, stop_early=True):
    fin = open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # n, d = map(int, fin.readline().split()) # We manually removed the first line
    data = {}
    i = 0

    for line in fin:
        # Skip Twitter specific words/tokens
        if re.search("^<[^>]+> ", line):
            continue
        key, values = load_fasttext_line(line)
        data[key] = values
        if i % 10000 == 0:
            print(i)
        if stop_early and i == 300000:  # 300k
            break
        i += 1

    return data


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


def format_qrel_line(target, qid, features, docid):
    result = f'{target} qid:{qid} '
    values = list(features.values())

    for i in range(0, len(values)):
        result += f'{i + 1}:{values[i]} '

    return result + f'#{docid}\n'


def compute_features(index_reader, query, docid):
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

    return {
        'bm25_score': "{:.8f}".format(bm25_score),
        'tf_idf_sum': "{:.8f}".format(tf_idf_sum),
        'total_terms': total_terms,
        'jm': "{:.8f}".format(jm),
        'dirich': "{:.8f}".format(dirich),
    }
