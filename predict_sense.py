'''
Given an unlabeled sentence, predict
the word sense of each token
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import collections
import itertools
import cPickle as pickle

from defs import EMBED_SIZE, VOCAB_SIZE


def read_file(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
    return content 


def load_word_vector_mapping(vocab_file, vector_file):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    Adopted from assignment 3
    """
    ret = collections.OrderedDict()
    for vocab, vector in zip(vocab_file, vector_file):
        vocab = vocab.strip()
        print(vocab)
        vector = vector.strip()
        print(vector)
        ret[vocab] = np.array(list(map(float, vector.split())))
    return ret


def load_embeddings(vocab_file, wordvec_file, string_to_index):
    """
    Load word vector embeddings
    Adopted from assignment 3
    """
    embeddings = np.array(np.random.randn(VOCAB_SIZE, EMBED_SIZE), dtype=np.float32)
    embeddings[0] = 0.0
    for word, vec in load_word_vector_mapping(vocab_file, wordvec_file).items():
        # see if word is in vocab; it will because of how we structured our stuff
        embeddings[string_to_index[word]] = vec
    print("Initialized embeddings.")
    return embeddings


def generate_dicts(reverse_dictionary):
    '''
    Takes in a reverse_dictionary (index -> string)
    and returns a dict string_to_index (to allow indexing
    into the weights matrix) and possible_senses, which
    is a default dict from word -> set of possible sense tags
    These are returned as a tuple
    '''
    string_to_index = {reverse_dictionary[key]:key for key in reverse_dictionary}
    possible_senses = collections.defaultdict(set)
    for word in string_to_index:
        without_tag = word.split('/')[0]
        possible_senses[without_tag].add(word)
    return string_to_index, possible_senses

def occurence_prob(window, string_to_index, embeddings):
    '''
    Given a window (as a tuple) of sense labeled words
    compute the probability of them occuring together
    '''
    windowsz = len(window)
    context = range(0, windowsz/2) + range(windowsz/2 + 2, windowsz)
    center_idx = string_to_index[window[windowsz/2 + 1]]
    probs = tf.log(tf.nn.softmax(embeddings[center_idx], embeddings))
    s = 0
    for i in context:
        s += probs[i]
    return s

def predict_sense(window, string_to_index, possible_senses, embeddings):
    '''
    Predicts the word sense of the
    center word in the window
    (assumes odd total window size)
    '''
    possibilities = [p for p in itertools.product(*[possible_senses[word] for word in window])]
    return max(possibilities, key=lambda x: occurence_prob(x))


with open("data/reverse_dictionary.p", "rb") as f:
    rd = pickle.load(f)

vocab = read_file("data/vocab.txt")
word_vectors = read_file("data/word_vectors.txt")
si, ps = generate_dicts(rd)
load_embeddings(vocab, word_vectors, si)
