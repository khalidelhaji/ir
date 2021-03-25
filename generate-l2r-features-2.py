import math
import random
import json
from pyserini.index import IndexReader
import io
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
import gensim
from gensim.models import FastText
from gensim.test.utils import datapath


TOTAL_DOCS = 8841823
LAMBDA = 0.5 # LMIR.JM hyperparameter
MU = 2000 # LMIR.DIR hyperparameter

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

def compute_features(index_reader, query, qid, docid, target):
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
			df = counts[0] # document frequency
			cf = counts[1] # collection frequency

			# TF-IDF
			idf = math.log(math.e, (TOTAL_DOCS / max(df, 1)))
			tf_idf_sum += (tf[word] / total_terms) * idf

			# Jelineck-Mercer (JM) smoothing (LMIR.JM)
			p_ml = tf[word] / total_terms
			p_global = tf[word] / max(cf, 1)
			jm *= (1 - LAMBDA) * p_ml + LAMBDA * p_global

			# Dirichlet smoothing (LMIR.DIR), the probabilites exceed 1 here for some reason
			dirich *= (tf[word] + MU * cf) / (total_terms + MU)

	bm25_score = "{:.8f}".format(bm25_score)
	tf_idf_sum = "{:.8f}".format(tf_idf_sum)
	jm = "{:.8f}".format(jm)
	dirich = "{:.8f}".format(dirich)
	line_format = f"{target} qid:{qid} 1:{bm25_score} 2:{tf_idf_sum} 3:{total_terms} 4:{jm} 5:{dirich} #{docid}\n"
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
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        #print(data[tokens[0]])
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
	#stop_words = set(stopwords.words('english'))
	#words = [w for w in words if not w in stop_words]
	
	return words

def main(queries_file, qrels_file, output_file, separator, write_negative):
	queries = read_topics(queries_file)
	index_reader = IndexReader('indexes/msmarco-passage')
	document_count = int(index_reader.stats()['documents'])
	qrels = open(qrels_file, 'r')

	#vectors = load_vectors("wiki-news-300d-1M.vec")

	#load_vectors("wiki-news-300d-1M.vec")

	#model = FastText.load_fasttext_format('wiki-news-300d-1M.bin')
	cap_path = datapath("/home/khalid/new2/ir/wiki-news-300d-1M-subword.bin")
	model = gensim.models.fasttext.load_facebook_vectors(cap_path)

	print("Done.")

	#with open(output_file, 'w') as output_file_handle:
	for line in qrels:
		line = line.strip().split(separator)

		qid = int(line[0])
		docid = line[2]
		target = line[3]
		query = queries[qid]['title']
		doc = json.loads(index_reader.doc_raw(docid))["contents"]

		# print(query)
		# print(doc)

		query_words = clean(query)
		doc_words = clean(doc)

		print(doc_words)

		for word in doc_words:

			print(model[word])

		return
		#print(doc_words)

	# 		output_file_handle.write(compute_features(index_reader, query, qid, docid, target))

	# 		# The evaluation set doesn't need negative examples.
	# 		if write_negative:
	# 			negative_docid = str(get_negative_docid(document_count, docid))
	# 			output_file_handle.write(compute_features(index_reader, query, qid, negative_docid, 0))


if __name__ == '__main__':
	# Uncommented by default to avoid running a long command (~20 minutes or so)
	# main('queries.train.tsv', 'qrels.train.tsv', 'ranklib-features/data_ranklib-train.txt', '\t', True)
	main('msmarco-test2019-queries.tsv', 'runs/run.msmarco-test2019-queries-bm25.trec', 'ranklib-features/data_ranklib-test.txt', ' ', False)