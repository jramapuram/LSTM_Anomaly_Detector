__author__ = 'jramapuram'
import os.path
import numpy as np
import matplotlib.pyplot as plt

from config import config
from random import randint
from math import sin, pi
from itertools import islice
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

def build_autoencoder(conf):
    model = Sequential()
    model.add(Dense(conf['input_dim']
                    , conf['hidden_dim']
                    , init=conf['initialization']
                    , activation=conf['activation']))
    model.add(Dropout(0.5))
    model.add(Dense(conf['hidden_dim']
                    , conf['input_dim']
                    , init=conf['initialization']
                    , activation=conf['activation']))
    #model.add(Activation(conf['activation']))
    model.compile(loss=conf['loss'], optimizer=conf['optimizer'])
    return model

def build_lstm_autoencoder(conf):
    model = Sequential()
    model.add(Embedding(conf['max_features'], conf['input_dim']))
    model.add(LSTM(conf['input_dim'], conf['hidden_dim']
                   , activation=conf['activation']
                   , inner_activation=conf['inner_init']
                   , truncate_gradient=conf['truncate_gradient']
                   , return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(conf['hidden_dim'], conf['input_dim']
                   , activation=conf['activation']
                   , inner_activation=conf['inner_init']
                   , truncate_gradient=conf['truncate_gradient']
                   , return_sequences=False))
    # model.add(Activation(conf['activation']))
    model.compile(loss=conf['loss'], optimizer=conf['optimizer'])
    return model

def elementwise_square(list):
    return [i ** 2 for i in list]

def generate_sin_wave(input_size, num_waves, offset=-1):
    delta = 2*pi/(input_size-offset)  # for proper shifting
    one_wave = [sin(delta*i) for i in xrange(0, input_size)]
    return one_wave*num_waves

# http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def add_amplitude_noise(conf, signal, num_errors=1):
    signal = np.array(signal)
    max_len = len(signal)
    for i in xrange(0, num_errors):
        location = randint(0, max_len - 1)
        signal[location] = signal[location] + np.random.normal(5, 0.1) #  XXX
        conf['noise_count'] = conf['noise_count'] + 1
    return signal

def split_vector(vec, split_size):
    return np.reshape(vec, (-1, split_size))

def plot_wave(wave, title='wave function'):
    fig = plt.figure()
    plt.plot(wave)
    fig.suptitle(title)

def load_model(path_str, model):
    if os.path.isfile(path_str):
        print 'model found, loading existing model...'
        model.load_weights(path_str)
        return True
    else:
        print 'model does not exist...'
        return False

def generate_data(full_length, window_length, periods):
    wave = generate_sin_wave(full_length, periods)
    generator = window(wave, window_length)
    return np.array([item for item in generator])

def generate_test_data(conf):
    test_data = generate_data(conf['input_dim']*conf['num_test_periods'], conf['input_dim'], 1)
    return np.matrix([add_amplitude_noise(conf, row, 1).flatten() if randint(0, 13) == 1
                    else row.flatten() for row in test_data])

if __name__ == "__main__":
    conf = config().get_config()

    print 'generating data....'
    X_train = generate_data(conf['input_dim'], conf['input_dim'], conf['num_periods'])
    print 'X_train size: %s | input_size: %d' % (X_train.shape, conf['input_dim'])
    X_test = generate_test_data(conf)
    print 'X_test size: %s | number of noise samples added: %d' % (X_test.shape, conf['noise_count'])

    print 'building lstm autoencoder...'
    model = build_lstm_autoencoder(conf)
    model.describe()
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

    # test_eval = [model.evaluate(item, item, batch_size=1, show_accuracy=False, verbose=False) for item in X_test]
    predictions = [model.predict_proba(item, batch_size=1, verbose=False) for item in X_test]
    mse_prediction = [np.array(elementwise_square((xtrue - xpred).T)).flatten() for xtrue, xpred in zip(X_test, predictions)]
    mse_prediction = np.array(mse_prediction).flatten()

    print 'plotting results of anomaly detection...'
    plot_wave(mse_prediction, 'mse prediction on sliding window')
    plot_wave(np.ravel(X_test), 'test wave unrolled')
    X_test_mean = np.array([np.mean(row) for row in X_test])
    plot_wave(np.ravel(X_test_mean), 'test wave mean approx')

    # XXX: Fix with proper classical tests like grubbs, etc.
    print 'anomaly detector caught ~ %d anomalies' % len(mse_prediction[np.where(mse_prediction > 5)])
    plt.show()

