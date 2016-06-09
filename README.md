Weighted-Embeddings
=======================

A system for learning word weights, optimised for sentence-level vector similarity.

You can find model details in the following paper:

Sentence Similarity Measures for Fine-Grained Estimation of Topical Relevance in Learner Essays  
Marek Rei and Ronan Cummins  
*In Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications (BEA)*  
San Diego, United States, 2016  


The trained weights are in [weightedembeddings_word_weights.txt](https://github.com/marekrei/weighted-embeddings/blob/master/weightedembeddings_word_weights.txt)  
They are desgined to be used together with the 300-dimensional [word2vec](https://code.google.com/archive/p/word2vec/) vectors, pretrained on Google News:  
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing


Running
----------------------------

Implementation requires numpy and Theano.

To calculate IDF weights:

    python idf_weighting.py pretrained_embeddings_path plain_text_corpus_path output_weights_path

To calculate weights based on sentence similarity:

    python cosine_weighting.py epochs pretrained_embeddings_path plain_text_corpus_path output_weights_path

The implementation is not currently parallelised. It runs reasonably fast on the BNC (100M words), but for larger corpora a more efficiect version could be implemented.
