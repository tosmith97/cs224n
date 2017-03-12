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
    return 0.0 # TODO: compute this

def predict_sense(window, string_to_index, possible_senses, embeddings):
    '''
    Predicts the word sense of the
    center word in the window
    (assumes odd total window size)
    '''
    possibilities = [p for p in itertools.product(*[possible_senses[word] for word in window])]
    return max(possibilities, key=lambda x: occurence_prob(x))







