from gensim import models
import spacy
import numpy as np

word2vec = models.Word2Vec.load('word2vec.model')
word2vec.wv.save_word2vec_format('word2vec.bin')


nlp = spacy.load("en_core_web_lg", vectors=False)
rows, cols = 0, 0
for i, line in enumerate(open('word2vec.bin', 'r')):
    if i == 0:
        rows, cols = line.split()
        rows, cols = int(rows), int(cols)
        nlp.vocab.reset_vectors(shape=(rows, cols))
    else:
        word, *vec = line.split()
        vec = np.array([float(i) for i in vec])
        nlp.vocab.set_vector(word, vec)
        print(word)

nlp.to_disk('spacy_word2vec')