import math
import random
from pyserini.index import IndexReader

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

def main(queries_file, qrels_file, output_file, separator, write_negative):
	queries = read_topics(queries_file)
	index_reader = IndexReader('indexes/msmarco-passage')
	document_count = int(index_reader.stats()['documents'])
	qrels = open(qrels_file, 'r')

	with open(output_file, 'w') as output_file_handle:
		for line in qrels:
			line = line.strip().split(separator)

			qid = int(line[0])
			docid = line[2]
			target = line[3]
			query = queries[qid]['title']

			output_file_handle.write(compute_features(index_reader, query, qid, docid, target))

			# The evaluation set doesn't need negative examples.
			if write_negative:
				negative_docid = str(get_negative_docid(document_count, docid))
				output_file_handle.write(compute_features(index_reader, query, qid, negative_docid, 0))


if __name__ == '__main__':
	# Uncommented by default to avoid running a long command (~20 minutes or so)
	# main('queries.train.tsv', 'qrels.train.tsv', 'ranklib-features/data_ranklib-train.txt', '\t', True)
	main('msmarco-test2019-queries.tsv', 'runs/run.msmarco-test2019-queries-bm25.trec', 'ranklib-features/data_ranklib-test.txt', ' ', False)