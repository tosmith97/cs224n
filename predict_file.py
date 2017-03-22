from __future__ import print_function
import numpy as np
import tensorflow as tf
# based on https://github.com/hunkim/word-rnn-tensorflow
import argparse
import time
import os
from six.moves import cPickle
from rnn_utils import get_file_str
from utils import TextLoader
from model import Model
#from predict_sense import *
#from rnn_utils import *
import time

with open('possible_senses.p', 'rb') as f:
    word_to_pos_senses = cPickle.load(f)

kInitialHistorySize = 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=200,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--input', type=str, default='') # full path to input data file
    args = parser.parse_args()
    input_data = get_file_str(args.input)
    load_and_predict(args, input_data)

# vocab  is word -> index
# words is index -> word
# mod = model
def predict_file(mod, sess, words, vocab, initial_history, eval_data):
    '''
    Predicts one file at a time.
    mod: Model object as described in model.py
    sess: tensorflow session
    words: index -> word
    vocab: word -> index
    initial_history: string of space separated senses labeled words
    word_to_pos_senses: dict word (str) -> set of possible senses
    eval_data: rest of the file after intial_history (list of words)
    returns sense labeled list of words
    '''
    num = len(eval_data) # check -1 on this (OBOB?)
    state = sess.run(mod.cell.zero_state(1, tf.float32))
    print(len(vocab)) #debug
    for word in initial_history[:-1]: #.split()[:-1]:
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word,0)
        feed = {mod.input_data: x, mod.initial_state:state}
        [state] = sess.run([mod.final_state], feed)
    ret = initial_history #.split()
    #word = eval_data[0] # not initial_history[-1] 
    for n in range(num): # num, debug
        word = eval_data[n] if eval_data[n] in word_to_pos_senses else '!'
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word,0)
        feed = {mod.input_data: x, mod.initial_state:state}
        [probs, state] = sess.run([mod.probs, mod.final_state], feed)
        p = probs[0]
        possible_senses = word_to_pos_senses[word]
        #print (possible_senses)
        sense_idx = [vocab[s] for s, counts in possible_senses] 
        #print (sense_idx)
        if (len(sense_idx) == 0):
            pred = '!'
            ret.append(pred)
            continue
        slice_idx = np.argmax(p[sense_idx]) # only this slice
        #print (slice_idx)
        sample = sense_idx[slice_idx]
        #print (sample)
        pred = words[sample]
        ret.append(pred)
        # word = pred
    return ret

def report_accuracy(truth, predicted):
    assert(len(truth) == len(predicted))
    total = 0
    correct = 0
    correct_pred = []
    incorrect_pred = []
    for i in range(len(truth)): # debugging print statements
        #print ('actual: ' + truth[i])
        #print ('predicted: ' + predicted[i])
        if (len(word_to_pos_senses[truth[i]]) == 1):
            continue
        if (truth[i] == predicted[i]):
            correct += len(word_to_pos_senses[truth[i]])
            correct_pred.append(truth[i])
        else:
            incorrect_pred.append((truth[i], predicted[i]))
        total += len(word_to_pos_senses[truth[i]])
    print ('accuracy: ' + str(correct * 1.0 / total)) 
    print (predicted)
    with open('correct_incorrect' + str(time.time()) + '.p', 'wb') as f:
        cPickle.dump({'correct_pred': correct_pred, 'incorrect_pred':incorrect_pred}, f)

def get_mls_for_window(word_list):
    print ('initial window ' + str(word_list))
    #return [w.split('/')[0] for w in word_list] #debu
    result = []
    for w in word_list:
        pos_senses = word_to_pos_senses[w]
        if len(pos_senses) > 0:
            result.append(max(pos_senses, key=lambda x: x[1])[0]) 
        else:
            print ('not contained: ' + w)
            result.append('!')
    return result

def load_and_predict(args, input_data):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    mod = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            unlabeled_input_data = [w.split('/')[0] for w in input_data]
            initial = unlabeled_input_data[:kInitialHistorySize]
            initial_history = get_mls_for_window(initial)
            eval_data = unlabeled_input_data[kInitialHistorySize:]	
            results = predict_file(mod, sess, words, vocab, initial_history, eval_data)      
            report_accuracy(input_data, results)
            #print(model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick))

if __name__ == '__main__':
    main()
