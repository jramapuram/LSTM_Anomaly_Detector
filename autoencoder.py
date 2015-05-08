__author__ = 'jramapuram'

import os.path

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


class AutoEncoder:
    def __init__(self, conf, X_train, X_test=None, model_type='LSTM'):
        self.model_type = 'LSTM'
        self.conf = conf
        self.model = None

        if self.model_type == 'LSTM':
            self.model = self.build_lstm_autoencoder(self.conf)
        else:
            self.model = self.build_autoencoder(self.conf)

        self.model.describe()
        self.train_autoencoder(X_train, X_test)

    def train_autoencoder(self, X_train, X_test):
        model_structure = 'weights_%din_%dhid_%dbatch_%depochs.dat'
        model_name = model_structure % (int(self.conf['--input_dim'])
                                        , int(self.conf['--hidden_dim'])
                                        , int(self.conf['--batch_size'])
                                        , int(self.conf['--max_epochs']))
        model_exists = self.load_model(model_name, self.model)

        if not model_exists:
            print 'training new model...'
            self.model.fit(X_train, X_train
                           , batch_size=int(self.conf['--batch_size'])
                           , nb_epoch=int(self.conf['--max_epochs'])
                           , validation_split=float(self.conf['--max_epochs'])
                           , show_accuracy=True)
            print 'saving model to %s...' % model_name
            self.model.save_weights(model_name)

        if X_test is not None:
            test_eval = [self.model.evaluate(item, item, batch_size=1, show_accuracy=False, verbose=False) for item in X_test]
            print 'test evaluation: ', test_eval

    @staticmethod
    def build_autoencoder(conf):
        model = Sequential()
        model.add(Dense(int(conf['--input_dim'])
                        , int(conf['--hidden_dim'])
                        , init=conf['--initialization']
                        , activation=conf['--activation']))
        model.add(Dropout(0.5))
        model.add(Dense(int(conf['--hidden_dim'])
                        , int(conf['--input_dim'])
                        , init=conf['--initialization']
                        , activation=conf['--activation']))
        # model.add(Activation(conf['--activation']))
        model.compile(loss=conf['--loss'], optimizer=conf['--optimizer'])
        return model

    @staticmethod
    def build_lstm_autoencoder(conf):
        model = Sequential()
        model.add(Embedding(int(conf['--max_features']), int(conf['--input_dim'])))
        model.add(LSTM(int(conf['--input_dim']), int(conf['--hidden_dim'])
                       , activation=conf['--activation']
                       , inner_activation=conf['--inner_init']
                       , init=conf['--initialization']
                       , truncate_gradient=int(conf['--truncated_gradient'])
                       , return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(int(conf['--hidden_dim']), int(conf['--input_dim'])
                       , activation=conf['--activation']
                       , inner_activation=conf['--inner_init']
                       , init=conf['--initialization']
                       , truncate_gradient=int(int(conf['--truncated_gradient']))
                       , return_sequences=False))
        #  model.add(Activation(conf['--activation']))
        model.compile(loss=conf['--loss'], optimizer=conf['--optimizer'])
        return model

    def get_model(self):
        return self.model

    def get_model_type(self):
        return self.model_type

    @staticmethod
    def load_model(path_str, model):
        if os.path.isfile(path_str):
            print 'model found, loading existing model...'
            model.load_weights(path_str)
            return True
        else:
            print 'model does not exist...'
            return False
