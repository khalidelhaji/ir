import io
from nltk.tokenize import word_tokenize
import string
import pandas as pd

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    i = 0
    for line in fin:
        tokens = line.rstrip().split(' ')
        list(map(float, tokens[1:]))
        data[tokens[0]] = list(map(float, tokens[1:]))
        if i % 10000 == 0:
            print(i)
        if i == 10000:
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

def compute_mean(vectors, text):
    cleaned = clean(text)

    word_vectors = []
    for word in cleaned:
        if word in vectors:
            word_vectors.append(vectors[word])

    return pd.DataFrame(word_vectors).mean()

def process_file(vectors, input, output):
    input_file = open(input, 'r')
    count = 0

    with open(output, 'w') as output_file_handle:
        output_file_handle.write('0, 300\n')
        for line in input_file:
            line = line.strip().split('\t')
            text_id = line[0]
            text = line[1]

            mean = compute_mean(vectors, text)
            string_floats = [str(float) for float in mean.tolist()]
            concatenated_floats = ' '.join(string_floats)
            output_file_handle.write(f'{text_id} {concatenated_floats}\n')

            if count % 10000 == 0:
                print(count)
            if count == 1000:
                return
            count += 1

def main():
    print('Loading vectors')
    vectors = load_vectors('crawl-300d-2M.vec')
    print('Calculating queries')
    process_file(vectors, 'queries.train.tsv', 'embeddings/queries-embeddings.train.tsv')
    # print('Calculating docs')
    # process_file(vectors, 'collection.tsv', 'embeddings/collection-embeddings.tsv')


if __name__ == '__main__':
    main()