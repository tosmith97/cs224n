'''
LSTM-based model for WSD
prediction purposes
IMPORTANT: Run with Python 3 (maybe -- might need to change this)
'''
# adapted from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# embedding inspiration: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# compatible unpickling with python2: http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
import h5py
import numpy as np
from scipy.sparse import lil_matrix
from keras.datasets import imdb
from keras.models import Sequential
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

import predict_sense
from process_xml import get_dir_list
from eval_model.py import has_valid_senses

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

def predict_labeled(model, history, embeddings, vocab, string_to_index):
    '''
    Given an RNN model and a previous word history
    that has senses labeled, predict the next word 
    
    Params:
        model: the model
        history: sentence of sense-tagged words, represented as list of strings
    '''
    # softmax will return |V|-vec, get index of highest val in vec
    # get word 
    
    # turn history into np array of wordvecs
    history = np.asarray([embeddings[h] for h in history])
    pred_vec = model.predict(history)
    pred_word = vocab[np.argmax(pred_vec)] # idx in vocab list 
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
    
    # first window of 10 is exhaustive search
    first_ten = list(predict_sense.predict_sense(eval_data[:10], predict_sense.si, predict_sense.ps, embeddings))

    # after first window, feed history into LSTM
    #### WARNING: might break 
    history = first_ten

    for i, curr_word in enumerate(eval_data[10:])
        if len(history) > 10:
            history.pop(0)

        # check to see if curr_word has senses 
        if idx_of_senses[i + 10]: 
            num_senses = len(predict_sense.ps[curr_word])
            tot_score += num_senses

            actual = idx_of_senses[i + 10]
            pred = predict_labeled(model, history, embeddings, vocab, predict_sense.si)
            
            # compare
            score += tot_score if actual == pred else 0

            # add pred sense to history
            history.append(pred)
        
        else:
            history.append(curr_word)

    acc = tot_score / score
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

def build_and_train_model(X_train, y_train):
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

    embedding = Embedding(kTopWords, kEmbeddingVectorLength,
        weights=[emb_matrix], input_length=kMaxLength, trainable=False)
    model.add(embedding)
    model.add(Dropout(rate=0.2))
    model.add(LSTM(100)) # return_sequences
    model.add(Dropout(rate=0.2))
    model.add(Dense(kTopWords, activation='softmax')) # do we need a dense layer?
    #model.add(TimeDistributed(Dense(1)))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.001, decay=1e-6, momentum=1.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
#    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    #model.fit(X_train, y_train, batch_size=64) #nb_epoch?
    model.fit_generator(generator=batch_generator(X_train, y_train, 32, True), nb_epoch=4,
    steps_per_epoch=X_train.shape[0]//10)
    return model


def main():
    '''
    Defines and trains an RNN language model

    '''
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
    model = build_and_train_model(X_train, y_train)
    model.save('rnn_10window_5epoch_smallLR.h5')
    # need to test on some sense examples

    # load vocab file, represent as array of words
    vocab = predict_sense.vocab
    possible_senses = predict_sense.ps

    # evaluation data is list of strings
    eval_data = get_dir_list('one_file/')

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




if __name__ == '__main__':
    main()