import h5py
import argparse, collections, datetime
import numpy as np
import tensorflow as tf
from defs import EMBED_SIZE

from predict_sense import ps, embeddings, si, vocab, rd
from rnn_utils import *

import _pickle as pickle        
        # load model
        # predict prob dist over V
        
    

def get_prob_dist(model, sess, words, vocab, num=200, prime='first all'):
    state = sess.run(model.cell.zero_state(1, tf.float32))
    if not len(prime) or prime == " ":
        prime  = random.choice(list(vocab.keys()))    
    print (prime)
    for word in prime.split()[:-1]:
        print (word)
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word,0)
        feed = {model.input_data: x, model.initial_state:state}
        [state] = sess.run([model.final_state], feed)

    ret = prime
    word = prime.split()[-1]
    for n in range(num):
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word,0)
        feed = {model.input_data: x, model.initial_state:state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)
        p = probs[0]

        pred = words[sample]
        ret += ' ' + pred
        word = pred
    return ret



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

def predict_labeled(model, history, embeddings, vocab, string_to_index, next_word):
    '''
    Given an RNN model and a previous word history
    that has senses labeled, predict the next word 
    
    Params:
        model: the model
        history: sentence of sense-tagged words, represented as list of strings
    '''
    # softmax will return |V|-vec, get index of highest val for a sense in vec 
    # get word 
    
    # turn history into np array of indices
    history = np.asarray([string_to_index[h] if h in string_to_index else string_to_index['UNK'] for h in history.split()])

    pred_vec = model.predict(np.reshape(history, (1, 10))).T
    
    # get probabilities of senses for the word
    senses = [string_to_index[s] for s in ps[next_word]]

    potential_words = []
    for s in senses:
        potential_words.append(list(pred_vec[s-1,:]))

    # idx of best wordvec -> senses[idx]
    idx = np.argmax(np.asarray(potential_words)) 

    pred_word = rd[senses[idx]]
    return pred_word


def eval_model(model, embeddings, vocab, eval_data, idx_of_senses):
    '''
    Evaluates model on eval_data
    
    Params: 
        model: the model
        embeddings: matrix, emb[idx] -> wordvector
        vocab: list of vocabulary
        eval_data: 
    '''
    tot_score = 0.0
    score = 0.0

    window = ' '.join([data for data in eval_data[:10]])
    
    # Assign MLS to first ten words 
    first_ten = get_mls_for_window(window, ps)


    with tf.Session() as sess:


    # after first window, feed history into LSTM
    #### WARNING: might break 
    history = first_ten
    incorrect = []
    correct = []

    for i, curr_word in enumerate(eval_data[10:-1]):
        if len(history) > 10:
            history.pop(0)

        # check to see if curr_word has senses 
        if (i + 10) in idx_of_senses: 
            num_senses = len(ps[curr_word])
            tot_score += num_senses

            actual = idx_of_senses[i + 10]
            window = ' '.join([word for word in history])
            pred = predict_labeled(model, window, embeddings, vocab, si, eval_data[i + 10])

            # compare
            if actual == pred:
                score += num_senses
                correct.append(("Predicted: " + pred, "Actual:" + actual))
            else:
                incorrect.append(("Predicted: " + pred, "Actual:" + actual))

            # add pred sense to history
            history.append(pred)
        
        else:
            history.append(curr_word)

    acc = score / tot_score
    print("Total Available Score:", tot_score)
    print("Model's Score:", score)
    print("Accuracy:", acc)

    # with open('rnn_incorrect.txt', 'w') as f:
    #         f.write("Tested: " + str(tot_score))
    #         f.write("\tAccuracy:" + str(acc))
    #         f.write("\n")
    #         f.write('\n'.join('%s %s' % x for x in incorrect))

    # with open('rn_correct.txt', 'w') as f:
    #         f.write("Tested: " + str(tot_score))
    #         f.write("\tAccuracy:" + str(acc))
    #         f.write("\n")
    #         f.write('\n'.join('%s %s' % x for x in correct))

# def predict(model, unlabeled_history):
#     '''
#     Given an RNN model and a previous word history
#     that doesn't have senses labled, predict a probability
#     distribution over the next possible word as a np
#     array. (need to iterate over cartesian product
#     of senses)
#     '''
#     pass


def main():
    # load model
    with open(os.path.join('save/', 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    model = Model(saved_args, True)


    # load vocab file, represent as array of words
    vocab_list = vocab
    possible_senses = ps

    # evaluation data is list of strings
    eval_data = get_dir_list('small_eval/', get_file_str)

    # the indices of words with senses
    idx_of_sense = collections.defaultdict()
    # only put indices of words w/ 1> senses in defaultdict to minimize space complexity
    # want to pass words without senses to eval
    for i, w in enumerate(eval_data):
        if '/' in w:
            eval_data[i] = w.split('/', 1)[0]

            # ensure word has >1 senses
            if has_valid_senses(eval_data[i], possible_senses):   
                idx_of_sense[i] = w

    eval_model(model, embeddings, vocab_list, eval_data, idx_of_sense)