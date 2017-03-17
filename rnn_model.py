'''
LSTM-based model for WSD
prediction purposes
'''
# adapted from http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.core import Dropout

kEmbeddingVectorLength = 128
kTopWords = 40000
kMaxLength = 100 # tweak this

# use predict_proba

def main():
    model = Sequential()
    model.add(Embedding(kTopWords, kEmbeddingVectorLength, input_length=kMaxLength))
    model.add(Dropout(rate=0.2))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print(model.summary())
    #(X_train, y_train), (X_test, y_test) = load_data()


if __name__ == '__main__':
    main()
