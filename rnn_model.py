'''
LSTM-based model for WSD
prediction purposes
'''
# adapted from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dropout

kEmbeddingVectorLength = 64
kTopWords = 40000
kMaxLength = 100 # tweak this

# use predict_proba
# look at embedding intialization

def add_seq(seq, string_to_index, X_train_list, y_train_list, history_len):
    '''
    seq: sequence of strings
    string_to_index: dictionary from string to int 
    X_train_list: training history to be modified
    y_train_list: corresponding next word
    history_len: length of sequences fed into RNN to predict next word
    '''
    for i in range(len(seq) - history_len):
        X_train_list.append([string_to_index[s] for s in seq[i:i+history_len]])
        y_train_list.append(string_to_index[seq[i+history_len]])

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
        add_seq(seq, string_to_index, X_train_list, y_train_list)
    return np.array(X_train_list), np.array(y_train_list)

def predict_labeled(model, history):
    '''
    Given an RNN model and a previous word history
    that has senses labeled, predict a probability
    distribution over the next possible word as a np
    array
    '''
    pass

def predict(model, unlabeled_history):
    '''
    Given an RNN model and a previous word history
    that doesn't have senses labled, predict a probability
    distribution over the next possible word as a np
    array. (need to iterate over cartesian product
    of senses)
    '''
    pass

def main():
    '''
    Defines and trains an RNN language model
    the model attempts to prdeict the next word
    given 
    '''
    model = Sequential()
    model.add(Embedding(kTopWords, kEmbeddingVectorLength, input_length=kMaxLength))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    #model.add(Dense(kTopWords)) # do we need a dense layer?
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print(model.summary())
    #(X_train, y_train), (X_test, y_test) = load_data()


if __name__ == '__main__':
    main()
