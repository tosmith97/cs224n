'''
LSTM-based model for WSD
prediction purposes
IMPORTANT: Run with Python 3 (maybe -- might need to change this)
Run with --train True to train the model, --eval True --model_path [path] to evaluate the model
'''
# adapted from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# embedding inspiration: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# compatible unpickling with python2: http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
import h5py
import argparse, collections
import numpy as np
from scipy.sparse import lil_matrix
from keras.datasets import imdb
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dropout
from keras.optimizers import SGD
from defs import EMBED_SIZE

from predict_sense import ps, embeddings, si, vocab
from rnn_utils import *

import _pickle as pickle

kEmbeddingVectorLength = EMBED_SIZE
kTopWords = 19989
kMaxLength = 10 # tweak this
kEmbeddingPath = 'data/word_vectors.txt'
kTrainingPath = 'data/rnn_windows_size_10.p'
kNumTraining = 494887
kUnkIdx = 0
# use predict_proba
# look at embedding intialization

def batch_generator(X, y, batch_size, shuffle):
    # from https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/22567#129557
    number_of_batches = X.shape[0]//batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = y[batch_index].toarray()
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
    
def get_index(input_str, string_to_index):
    '''
    If input_str exists, give the corresponding index
    otherwise give the UNK index.
    '''
    return string_to_index[input_str] if input_str in string_to_index else string_to_index['UNK']

def tostr(train_row, index_to_string):
    '''
    Given a row from X_train (train_row) and a dictionary
    from indices to strings (reverse dictionary), convert
    the training example from a sequence of numbers to
    a sequence of strings.
    '''
    return ' '.join([index_to_string[i] for i in arr])

def add_seq(seq, string_to_index, X_train_list, y_train_list, history_len):
    '''
    seq: sequence of strings
    string_to_index: dictionary from string to int
    X_train_list: training history to be modified
    y_train_list: corresponding next word
    history_len: length of sequences fed into RNN to predict next word
    '''
    for i in range(len(seq) - history_len):
        X_train_list.append([get_index(s, string_to_index) for s in seq[i:i+history_len]])
        y_train_list.append(get_index(seq[i+history_len], string_to_index))

def prepare_data(all_sequences, string_to_index, history_len=5):
    '''
    Takes of every file's word sequence and returns
    a pair (X_train, y_train) in the following format:
    X_train: <word1> <word2> ... <wordn>
    y_train: <nextword> (after X_train)
    This data will then be fed into the RNN language model
    '''
    X_train_list = []
    y_train_list = []
    for seq in all_sequences:
        add_seq(seq, string_to_index, X_train_list, y_train_list, history_len)
    return np.array(X_train_list), np.array(y_train_list)

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
    
    # turn history into np array of wordvecs
    history = [string_to_index[h] if h in string_to_index else string_to_index['UNK'] for h in history.split()]
    history = np.asarray([embeddings[h] for h in history]).T
    pred_vec = model.predict_proba(history)
    print('L', len(pred_vec))

    # get probabilities of senses for the word
    senses = [string_to_index[s] for s in ps[next_word]]
    potential_words = [pred_vec[s] for s in senses]
    pred_word = vocab[np.argmax(pred_vec)] # idx in vocab list 
    print(pred_word)
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

    # Cartesian product TBD

    # after first window, feed history into LSTM
    #### WARNING: might break 
    history = first_ten

    for i, curr_word in enumerate(eval_data[10:-1]):
        if len(history) > 10:
            history.pop(0)

        # check to see if curr_word has senses 
        if (i + 10) in idx_of_senses: 
            num_senses = len(ps[curr_word])
            tot_score += num_senses

            actual = idx_of_senses[i + 10]
            window = ' '.join([word for word in history])
            pred = predict_labeled(model, window, embeddings, vocab, si, eval_data[i + 10 + 1])


            # compare
            score += tot_score if actual == pred else 0

            # add pred sense to history
            history.append(pred)
        
        else:
            history.append(curr_word)

    acc = score / tot_score
    print("Total Available Score:", tot_score)
    print("Model's Score:", score)
    print("Accuracy:", acc)

# def predict(model, unlabeled_history):
#     '''
#     Given an RNN model and a previous word history
#     that doesn't have senses labled, predict a probability
#     distribution over the next possible word as a np
#     array. (need to iterate over cartesian product
#     of senses)
#     '''
#     pass

def build_and_train_model(X_train, y_train, learn_embedding=False, learned_emb_dim=32):
    '''
    Builds keras LSTM model and trains using
    the provided input np arrays. the model
    attempts to predict the next word given
    a previous history (of labeled data)
    The trained model is returned
    '''
    model = Sequential() # can feed in our own embeddings
    emb_matrix = np.loadtxt(kEmbeddingPath)
    kTopWords = emb_matrix.shape[0]
    kEmbeddingVectorLength = emb_matrix.shape[1]

    if learn_embedding:
        embedding = Embedding(kTopWords, learned_emb_dim, input_length=kMaxLength, trainable=True)
    else:
        embedding = Embedding(kTopWords, kEmbeddingVectorLength,
            weights=[emb_matrix], input_length=kMaxLength, trainable=False)
    
    model.add(embedding)
    model.add(Dropout(rate=0.25))
    model.add(LSTM(100)) # return_sequences
    model.add(Dropout(rate=0.25))
    model.add(Dense(kTopWords, activation='softmax')) # do we need a dense layer?
    #model.add(TimeDistributed(Dense(1)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=1.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
#    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    #model.fit(X_train, y_train, batch_size=64) #nb_epoch?
    model.fit_generator(generator=batch_generator(X_train, y_train, 32, True), nb_epoch=10,
    steps_per_epoch=X_train.shape[0]//10)
    return model


def main():
    '''
    Defines and trains an RNN language model

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store', dest='train', type=bool, default=False)
    parser.add_argument('--eval', action='store', dest='eval', type=bool, default=False)
    parser.add_argument('--learn_embeddings', action='store', dest='learn_embeddings', type=bool, default=False)
    parser.add_argument('--model_path', action='store', dest='model_path', type=str, default='rnn_10window_5epoch_smallLR.h5')  

    results = parser.parse_args()
    train = results.train
    eval = results.eval     
    model_path = results.model_path
    learn_embeddings = results.learn_embeddings

    if train:
        # set stateful = True? use 1-hot representation of words?
        with open(kTrainingPath, 'rb') as f:
            training_dict = pickle.load(f, encoding='latin1')
            X_train, y_train = training_dict['X_train'], training_dict['y_train']
        #new_X = np.zeros(X_train.shape[0], dtype=list)
        #for i in range(X_train.shape[0]):
        #    new_X[i] = X_train[i].tolist()
        #X_train = new_X#y_train.reshape((kNumtraining,))
        X_train[X_train >= kTopWords] = kUnkIdx
        y_one_hot = lil_matrix((y_train.shape[0], kTopWords))
        for i in range(y_train.shape[0]):
            pos = y_train[i]
            pos = pos if pos < kTopWords else kUnkIdx
            y_one_hot[i, pos] = 1
        y_train = y_one_hot
        print('Shape of X_train: ' + str(X_train.shape))
        print('Shape of y_train: ' + str(y_train.shape))
        model = build_and_train_model(X_train, y_train, learn_embeddings)
        model.save('rnn_10window_5epoch_smallLR.h5')
        # need to test on some sense examples

    if eval:
        # load model
        model = load_model(model_path)

        # load vocab file, represent as array of words
        vocab_list = vocab
        possible_senses = ps

        # evaluation data is list of strings
        eval_data = get_dir_list('one_file/', get_file_str)

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


if __name__ == '__main__':
    main()