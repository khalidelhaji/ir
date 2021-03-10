from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
import random
import itertools
import math
from nltk import FreqDist
import nltk
import json

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

C_size = 50

index_reader = IndexReader('indexes/msmarco-passage')

top_25 = [{'term': 'you', 'cf': 3704969}, {'term': 'your', 'cf': 2871978}, {'term': 'from', 'cf': 2433977}, {'term': 'us', 'cf': 2215803}, {'term': 'can', 'cf': 2199793}, {'term': '1', 'cf': 1978831}, {'term': 'have', 'cf': 1794081}, {'term': 'â', 'cf': 1739619}, {'term': '2', 'cf': 1588930}, {'term': 'on', 'cf': 1303802}, {'term': 'which', 'cf': 1183799}, {'term': 'more', 'cf': 1148398}, {'term': 'ha', 'cf': 1140157}, {'term': 'i', 'cf': 1128679}, {'term': 'year', 'cf': 1104284}, {'term': 's', 'cf': 1090200}, {'term': '3', 'cf': 1087187}, {'term': 'all', 'cf': 1027818}, {'term': 'other', 'cf': 1022598}, {'term': 'when', 'cf': 1005864}, {'term': 'time', 'cf': 1005023}, {'term': 'also', 'cf': 991428}, {'term': 'mai', 'cf': 955836}, {'term': 'most', 'cf': 927327}, {'term': 'about', 'cf': 909980}]

total_words = index_reader.stats()['total_terms']

def dirich(freq_term_in_doc, total_words_in_doc, freq_term_in_collection, total_words, mu=1000, log=True):
    output = 0

    if log:
        output = math.log((freq_term_in_doc + mu*(freq_term_in_collection / total_words)) / (total_words_in_doc + mu))
    else:
        output = (freq_term_in_doc + mu*(freq_term_in_collection / total_words)) / (total_words_in_doc + mu)

    return max(0, output)


def relvance_model_prob(C, term, freq_term_in_collection):
    prob_word_given_relevance = 0

    for doc in C:
        content = json.loads(index_reader.doc_raw(doc['doc_id']))
        content = content['contents']

        tokens_doc = nltk.tokenize.word_tokenize(content)
        tf = FreqDist(tokens_doc)

        total_words_in_doc = len(tokens_doc)

        if term in tf:
            freq_term_in_doc = tf[term]
        else:
            freq_term_in_doc = 1

        prob_word_given_relevance += dirich(freq_term_in_doc, total_words_in_doc, \
            freq_term_in_collection, total_words, log=False)*doc['qld_score']

    return prob_word_given_relevance



def prf(query, data, id, output_file_handle):
    data = sorted(data, key=lambda k: k['qld_score'], reverse=True) 

    C = data[:C_size]

    score = []

    values_relevance_model = {}

    for top_word in top_25:
        freq_term_in_collection = top_word['cf']
        top_word = top_word['term']

        values_relevance_model[top_word] = relvance_model_prob(C, top_word, freq_term_in_collection)

    for doc in data:
        content = json.loads(index_reader.doc_raw(doc['doc_id']))
        content = content['contents']

        tokens_doc = nltk.tokenize.word_tokenize(content)
        freq_doc = FreqDist(tokens_doc)

        total_words_in_doc = len(tokens_doc)

        doc_rank = 0

        for top_word in top_25:
            freq_term_in_collection = top_word['cf']
            top_word = top_word['term']

            if top_word in freq_doc:
                freq_term_in_doc = freq_doc[top_word]
            else:
                freq_term_in_doc = 1

            # tokens_query = nltk.tokenize.word_tokenize(query)
     
            # f_query_words = FreqDist(tokens_query)

            # if top_word in tokens_query:
            #     p_w_given_query = f_query_words[top_word] / len(tokens_query)
            # else:
            #     p_w_given_query = 1 / len(tokens_query)

            mixture = values_relevance_model[top_word]

            doc_rank += mixture*dirich(freq_term_in_doc, total_words_in_doc, \
                        freq_term_in_collection, total_words, log=False)

        doc['relevance_rank'] = doc_rank

    reranked = sorted(data, key=lambda k: k['relevance_rank'], reverse=True) 

    for i in range(0, len(reranked)):
        _ = output_file_handle.write('{} Q0 {} {} {:.6f} AnseriniMOD\n'.format(id, reranked[i]['doc_id'], i+1, reranked[i]['relevance_rank']))



def run_all_queries(output_file_name, topics, searcher):
    with open(output_file_name, 'w') as output_file_handle:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in topics:
            query = topics[id]['title']
            hits = searcher.search(query, 1000)

            qid_docid_matches = []

            for i in range(0, len(hits)):
                data_object = {
                    "query_id": id,
                    "index": i+1,
                    "doc_id": hits[i].docid,
                    "qld_score": hits[i].score
                }

                qid_docid_matches.append(data_object)

            prf(topics[id]['title'], qid_docid_matches, id, output_file_handle)

            cnt += 1
            if cnt % 100 == 0:
                print(f'{cnt} queries completed')

def main():
    # This assumes the index has already been generated
    searcher = SimpleSearcher('indexes/msmarco-passage')
    searcher.set_qld()

    topics = read_topics('msmarco-test2019-queries.tsv')

    run_all_queries('runs/run.msmarco-test2019-queries-bm25.trec', topics, searcher)

if __name__ == '__main__':
    main()