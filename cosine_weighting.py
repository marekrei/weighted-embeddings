import sys
import collections
import numpy
import math
import theano

floatX=theano.config.floatX

class SentenceSum(object):
    def __init__(self, embeddings):
        sentence_ids = theano.tensor.ivector('sentence')
        neighbour_ids = theano.tensor.ivector('neighbour')
        adversary_ids = theano.tensor.ivector('adversary')
        learningrate = theano.tensor.fscalar('learningrate')

        self.word_weights = theano.shared(numpy.ones((embeddings.shape[0],), dtype=floatX), 'word_weights')
        embeddings = theano.shared(embeddings, 'embeddings')

        def construct_vector(ids):
            vectors = embeddings[ids]
            vector = theano.tensor.dot(self.word_weights[ids].T, vectors)
            vector = vector / vector.norm(2)
            return vector

        sentence_vector = construct_vector(sentence_ids)
        neighbour_vector = construct_vector(neighbour_ids)
        adversary_vector = construct_vector(adversary_ids)

        cost = theano.tensor.maximum(theano.tensor.dot(sentence_vector, adversary_vector) - theano.tensor.dot(sentence_vector, neighbour_vector), 0.0)

        params = [self.word_weights,]
        gradients = theano.tensor.grad(cost, params, disconnected_inputs='warn')
        updates = [(p, p - (learningrate * g)) for p, g in zip(params, gradients)]
        self.train = theano.function([sentence_ids, neighbour_ids, adversary_ids, learningrate], [cost,], updates=updates, on_unused_input='warn', allow_input_downcast = True)


def sentence_to_ids(sentence, word2id):
    ids = []
    for word in sentence.strip().split():
        if word in word2id:
            ids.append(word2id[word])
    return numpy.array(ids, dtype=numpy.int32)


def create_cosine_weights(embeddings_path, corpus_path, learningrate=0.1, epochs=500, datapoints_per_epoch=10000, neighbour_scale=2.5):
    sentences = []
    with open(corpus_path, 'r') as f:
        for line in f:
            sentences.append(line.strip())

    word2id = collections.OrderedDict()
    embeddings = None
    with open(embeddings_path, 'r') as f:
        line_parts = f.next().strip().split()
        embeddings = numpy.zeros((100000, int(line_parts[1])), dtype=floatX)
        for line in f:
            #if "_" in line: #skipping phrase embeddings
            #    continue
            line_parts = line.strip().split()
            word_id = len(word2id)
            word2id[line_parts[0]] = word_id
            if word_id >= embeddings.shape[0]:
                embeddings = numpy.concatenate((embeddings, numpy.zeros((100000, embeddings.shape[1]), dtype=floatX)), axis=0)
            embeddings[word_id] = numpy.array([float(v) for v in line_parts[1:]])
    embeddings = embeddings[:len(word2id)]

    sentencesum = SentenceSum(embeddings)
    numpy.random.seed(1)
    for epoch in xrange(epochs):
        print "epoch: " + str(epoch)
        cost_sum = 0.0
        count = 0
        for i in xrange(datapoints_per_epoch):
            position = numpy.random.randint(0, len(sentences)-1)
            neighbour = position
            while neighbour == position or neighbour < 0 or neighbour >= len(sentences):
                neighbour = position + int(round(numpy.random.normal(loc=0.0, scale=neighbour_scale)))
            adversary = numpy.random.randint(0, len(sentences)-1)
            sentence_ids = sentence_to_ids(sentences[position], word2id)
            neighbour_ids = sentence_to_ids(sentences[neighbour], word2id)
            adversary_ids = sentence_to_ids(sentences[adversary], word2id)
            if len(sentence_ids) == 0 or len(neighbour_ids) == 0 or len(adversary_ids) == 0:
                continue
            cost, = sentencesum.train(sentence_ids, neighbour_ids, adversary_ids, learningrate)
            if math.isnan(cost):
                sys.exit(1)
            cost_sum += cost
            count += 1
        print "average_cost: " + str(cost_sum / float(count))

    model_weights = sentencesum.word_weights.get_value()
    weights = collections.OrderedDict()
    for word in word2id:
        weights[word] = model_weights[word2id[word]]

    return weights


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    embeddings_path = sys.argv[2]
    corpus_path = sys.argv[3]
    output_path = sys.argv[4]

    weights = create_cosine_weights(embeddings_path, corpus_path, epochs=epochs)

    with open(output_path, 'w') as f:
        for word in weights:
            f.write(word + "\t" + str(weights[word]) + "\n")

