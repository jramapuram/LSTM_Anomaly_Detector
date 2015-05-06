__author__ = 'jramapuram'
import numpy as np
from config import config
from math import sin, pi
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

def build_autoencoder(conf):
    model = Sequential()
    model.add(Embedding(conf['max_features'], conf['input_dim']))
    model.add(LSTM(conf['input_dim'], conf['hidden_dim']
                   , activation=conf['activation']
                   , inner_activation=conf['inner_init']))
    model.add(Dropout(0.5))
    model.add(Dense(conf['hidden_dim'], conf['input_dim'], init=conf['initialization']))
    model.add(Activation(conf['activation']))

    model.compile(loss=conf['loss'], optimizer=conf['optimizer'])
    return model

def generate_sin_wave(input_size, num_waves):
    delta = 2*pi/(input_size-1)  # for proper shifting
    one_wave = [sin(delta*i) for i in xrange(0, input_size)]
    return one_wave*num_waves

def split_vector(vec, split_size):
    return np.reshape(vec, (-1, split_size))

def test_sin_wave(conf):
    plt.plot(generate_sin_wave(conf['input_dim'], conf['num_periods']))
    plt.show()

if __name__ == "__main__":
    conf = config().get_config()
    X_train = split_vector(generate_sin_wave(conf['input_dim'], conf['num_periods']), conf['input_dim'])
    print 'X_train size: %s | input_size: %d' % (X_train.shape, conf['input_dim'])
    X_test = split_vector(generate_sin_wave(conf['input_dim'], conf['num_test_periods']), conf['input_dim'])
    print 'X_test size: %s | input_size: %d' % (X_test.shape, conf['input_dim'])

    model = build_autoencoder(conf)
    model.fit(X_train, X_train, batch_size=16, nb_epoch=100, validation_split=0.1, show_accuracy=True)
    score = model.evaluate(X_test, X_test, batch_size=16)
