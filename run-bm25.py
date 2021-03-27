from pyserini.search import SimpleSearcher

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

def run_all_queries(output_file_name, topics, searcher):
    with open(output_file_name, 'w') as output_file_handle:
        cnt = 0
        print('Running {} queries in total'.format(len(topics)))
        for id in topics:
            query = topics[id]['title']
            hits = searcher.search(query, 1000)
            for i in range(0, len(hits)):
                _ = output_file_handle.write('{}\tQ0\t{}\t{}\t{:.6f}\tAnserini\n'.format(id, hits[i].docid, i+1, hits[i].score))
            cnt += 1
            if cnt % 100 == 0:
                print(f'{cnt} queries completed')

def main():
    # This assumes the index has already been generated
    searcher = SimpleSearcher('indexes/msmarco-passage')
    # searcher.set_bm25(0.82, 0.68)

    topics = read_topics('msmarco-test2019-queries.tsv')

    run_all_queries('runs/run.msmarco-test2019-queries-bm25.trec', topics, searcher)

if __name__ == '__main__':
    main()