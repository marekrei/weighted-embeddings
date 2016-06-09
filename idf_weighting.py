import sys
import math
import collections

def create_idf_weights(embeddings_path, corpus_path):
    df = {}
    document_count = 0
    with open(corpus_path) as f:
        for document in f:
            words = set(document.strip().split())
            for word in words:
                if word not in df:
                    df[word] = 0.0
                df[word] += 1.0
            document_count += 1

    idf = collections.OrderedDict()
    with open(embeddings_path) as f:
        for line in f:
            line_parts = line.strip().split()
            if len(line_parts) <= 2:
                continue
            word = line_parts[0]
            n = 1.0
            if word in df:
                n += df[word]
            score = math.log(document_count / float(n))
            idf[word] = score
    return idf


if __name__ == "__main__":
    embeddings_path = sys.argv[1]
    corpus_path = sys.argv[2]
    output_path = sys.argv[3]

    weights = create_idf_weights(embeddings_path, corpus_path)

    with open(output_path, 'w') as f:
        for word in weights:
            f.write(word + "\t" + str(weights[word]) + "\n")
