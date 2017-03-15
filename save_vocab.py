import cPickle as pickle
import numpy as np

with open("data/reverse_dictionary.p", "rb") as f:
    rd = pickle.load(f)

# save vocab
vocab = rd.values()
bad_indices = []

with open('data/vocab.txt', 'w') as f:
    for i, v in enumerate(vocab):
      try:
        f.write("%s\n" % v)
      except UnicodeEncodeError:
        bad_indices.append(i)

wordvecs = np.loadtxt('data/word_vectors.txt').tolist()

for idx in bad_indices:
    wordvecs.pop(idx)

wordvecs = np.asarray(wordvecs)
np.savetxt('data/word_vectors.txt', wordvecs)


