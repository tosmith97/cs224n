import xml.etree.ElementTree
import sys
import collections
import os
import fnmatch
import pickle
import tensorflow as tf
import time
import argparse

from process_xml import get_dir_list, is_valid, is_apostrophe_chunk, get_sense_name, is_number
import predict_sense
from defs import WINDOW_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname', action='store', dest='dirname', type=str)  
    parser.add_argument('--incorr_fn', action='store', dest='incorr_fn', type=str, default='')
    parser.add_argument('--correct_fn', action='store', dest='correct_fn', type=str, default='')
    parser.add_argument('--weight_scores', action='store', dest='weight_scores', type=bool, default=False)

    # MLS = Most Likeley Sense
    parser.add_argument('--mls', action='store', dest='mls', type=bool, default=False)


    results = parser.parse_args()
    dirname = results.dirname
    incorr_fn = results.incorr_fn
    correct_fn = results.correct_fn
    weight_scores = results.weight_scores
    mls = results.mls

    eval_data = get_dir_list(dirname, get_file_str)
    idx_of_sense = collections.defaultdict()

    possible_senses = predict_sense.ps

    # only put indices of words w/ 1> senses in defaultdict to minimize space complexity
    # want to pass words without senses to eval
    for i, w in enumerate(eval_data):
        if '/' in w:
            eval_data[i] = w.split('/', 1)[0]

            # ensure word has >1 senses
            if has_valid_senses(eval_data[i], possible_senses):   
                idx_of_sense[i] = w
    
    see_prediction_results(eval_data, idx_of_sense, incorr_fn, correct_fn, weight_scores, mls)


def has_valid_senses(word, possible_senses):
    ''' 
    We want words with >1 unique senses
    It is okay for our model to differentiate between an ambiguous sense (word) and a defined sense (word1/sense1)
    E.g. Good: word1 -> word1/sense1, word1/sense2, ...
               word1 -> word1, word1/sense1

         Bad: word1 -> word1
              word1 -> word1/sense1
    '''

    return not len(possible_senses[word]) < 2


def get_file_str(filename):
    root = xml.etree.ElementTree.parse(filename).getroot()

    # returns list of words for easier processing
    return get_word_sequence(root)


def process_element(element):
    '''
    tag a token with its sense
    '''
    suffix = element.get('sense')
    if suffix is None:
        suffix = ''
    return element.get('text') + suffix


def get_word_sequence(root):
    '''
    Iterate through xml root 
    and compile everything into
    a list of strings. Handle apostrophes
    by gluing together to form 
    contractions.
    '''
    seq = [get_sense_name(word) for word in root]
    fixed_seq = []
    for i in xrange(len(seq) - 1):
        if is_apostrophe_chunk(seq[i]):
            continue
        to_append = seq[i]
        if is_apostrophe_chunk(seq[i+1]) and 'sense' not in seq[i]:
            to_append += seq[i+1]
        fixed_seq.append(to_append)
    if not is_apostrophe_chunk(seq[-1]) and len(seq[-1]) > 0:
        fixed_seq.append(seq[-1])

    fixed_seq = [word.lower() for word in fixed_seq if is_valid(word)]
    for i, s in enumerate(fixed_seq):
        if is_number(s):
            fixed_seq[i] = 'NUM'

    return fixed_seq


def window_has_senses(idx_of_sense, idx_range):
    # idx_range is tuple, idx_of_sense is dict w/ all the stuff
    return any(key in idx_of_sense for key in range(idx_range[0], idx_range[1]))


def get_senses_in_window(window, idx_of_sense, idx_range):
    window_senses = []
    
    for idx in range(idx_range[0], idx_range[1]):
        if idx in idx_of_sense:
            window_senses.append(idx_of_sense[idx])
        else:
            window_senses.append(None)
    
    return window_senses


def get_mls_for_window(window, possible_senses):
    mls_window = []

    for word in window.split():
        senses = list(possible_senses[word])

        if len(senses) > 0:
            most_common_sense = senses[0]
        else:
            most_common_sense = None
        mls_window.append(most_common_sense)

    return mls_window


def see_prediction_results(eval_data, idx_of_sense, incorr_fn, correct_fn, weighted_scores, mls):
    # take window
    # See if words in window have senses that need predicting 
    # predict senses for WINDOW_SIZE
    # compare predicted senses to actual senses
    # have acc = num_correct/tot
    with tf.device('/gpu:0'):
        num_correct = 0.0
        tot_comparisons = 0.0
        incorrect = []
        correct = []

        embeddings = predict_sense.embeddings
        string_to_index = predict_sense.si
        possible_senses = predict_sense.ps

        start = time.time()
        # Current predict_sense finds predictions for whole window - do we need a center word? 
        for i in range(WINDOW_SIZE, len(eval_data) - WINDOW_SIZE):
            if window_has_senses(idx_of_sense, (i - WINDOW_SIZE, i + WINDOW_SIZE + 1)):
                # predict that shit
                window = ' '.join(eval_data[i - WINDOW_SIZE: i + WINDOW_SIZE + 1])           
                pred_senses = get_mls_for_window(window, possible_senses) if mls else list(predict_sense.predict_sense(window, string_to_index, possible_senses, embeddings))
                senses = get_senses_in_window(window, idx_of_sense, (i - WINDOW_SIZE, i + WINDOW_SIZE + 1))

                center_pred = pred_senses[WINDOW_SIZE]
                num_senses = len(possible_senses[eval_data[i]])

                if senses[WINDOW_SIZE] is not None and center_pred is not None:
                    tot_comparisons += num_senses
                    if senses[WINDOW_SIZE] == center_pred:
                        num_correct += num_senses
                    else:
                        print 'center_pred', center_pred
                        print 'actual', senses[WINDOW_SIZE]
                        print

        print("Evaluation took %.2f seconds" % (time.time() - start))
        print("Accuracy:", num_correct/tot_comparisons)
        with open(incorr_fn, 'w') as f:
            f.write("Tested: " + str(tot_comparisons))
            f.write("\tAccuracy:" + str(float(num_correct)/tot_comparisons))
            f.write("\n")
            f.write('\n'.join('%s %s' % x for x in incorrect))

        with open(correct_fn, 'w') as f:
            f.write("Tested: " + str(tot_comparisons))
            f.write("\tAccuracy:" + str(float(num_correct)/tot_comparisons))
            f.write("\n")
            f.write('\n'.join('%s %s' % x for x in correct))

if __name__ == '__main__':
    main()
