__author__ = 'jramapuram'
import os.path
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

def generate_sin_wave(input_size, num_waves, offset=-1):
    delta = 2*pi/(input_size-offset)  # for proper shifting
    one_wave = [sin(delta*i) for i in xrange(0, input_size)]
    return one_wave*num_waves

def split_vector(vec, split_size):
    return np.reshape(vec, (-1, split_size))

def test_sin_wave(conf):
    plt.plot(generate_sin_wave(conf['input_dim'], conf['num_periods']))
    plt.show()

def load_model(path_str, model):
    if os.path.isfile(path_str):
        print 'model found, loading existing model...'
        model.load_weights(path_str)
        return True
    else:
        print 'model does not exist...'
        return False

if __name__ == "__main__":
    conf = config().get_config()

    print 'generating data....'
    X_train = split_vector(generate_sin_wave(conf['input_dim'], conf['num_periods']), conf['input_dim'])
    print 'X_train size: %s | input_size: %d' % (X_train.shape, conf['input_dim'])
    X_test = split_vector(generate_sin_wave(conf['input_dim'], conf['num_test_periods']), conf['input_dim'])
    print 'X_test size: %s | input_size: %d' % (X_test.shape, conf['input_dim'])

    print 'building lstm autoencoder...'
    model = build_autoencoder(conf)
    model_name = conf['model_file'] % (conf['input_dim'], conf['hidden_dim'], conf['batch_size'], conf['max_epochs'])
    model_exists = load_model(model_name, model)

    if not model_exists:
        print 'training new model...'
        model.fit(X_train, X_train
                  , batch_size=conf['batch_size']
                  , nb_epoch=conf['max_epochs']
                  , validation_split=0.1
                  , show_accuracy=True)
        print 'saving model to %s...' % model_name
        model.save_weights(model_name)

    score = model.evaluate(X_test, X_test, batch_size=conf['batch_size'], show_accuracy=True)
    print 'model score on test set: ', score

