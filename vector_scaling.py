import sys
import numpy

if __name__ == "__main__":
    embeddings_path = sys.argv[1]
    weights_path = sys.argv[2]
    output_path = sys.argv[3]

    weights = {}
    with open(weights_path, 'r') as f:
        for line in f:
            line_parts = line.strip().split()
            assert(len(line_parts) == 2), "Incorrect weight format"
            weights[line_parts[0]] = float(line_parts[1])

    with open(embeddings_path, 'r') as f:
        with open(output_path, 'w') as o:
            for line in f:
                line_parts = line.strip().split()
                if len(line_parts) <= 2:
                    o.write(line.strip() + "\n")
                    continue
                vector = numpy.array([float(val) for val in line_parts[1:]])
                word = line_parts[0]
                if word in weights:
                    vector = weights[word] * vector
                    o.write(word)
                    for val in vector:
                        o.write(" " + str("{:.8f}".format(val)))
                    o.write("\n")
