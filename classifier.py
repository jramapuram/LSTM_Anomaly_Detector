__author__ = 'jramapuram'

import os.path

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


class Classifier:
    def __init__(self, classifier_type, conf):
        self.conf = conf
        self.model_type = classifier_type
        self.model = Sequential()
        if self.model_type.strip().lower() == 'lstm':
            self.model.add(Embedding(int(self.conf['--max_features'])
                                     , int(self.conf['--input_dim'])))

    def train_classifier(self, X_train, Y_train):
        self.model.get_config(verbose=1)
        self.model.compile(loss='binary_crossentropy'
                           , optimizer=self.conf['--optimizer'])

        model_structure = 'weights_%din_%dout_%dbatch_%depochs_%s_classifier.dat'
        model_name = model_structure % (int(self.conf['--input_dim'])
                                        , 1
                                        , int(self.conf['--batch_size'])
                                        , int(self.conf['--max_epochs_classifier'])
                                        , self.model_type)
        model_exists = self.load_model(model_name, self.model)

        if not model_exists:
            print 'training new model using %s loss function & %s optimizer...' \
                  % ('binary_crossentropy', self.conf['--optimizer'])

            self.model.fit(X_train, Y_train
                           , batch_size=int(self.conf['--batch_size'])
                           , nb_epoch=int(self.conf['--max_epochs_classifier'])
                           , validation_split=float(self.conf['--validation_ratio'])
                           , show_accuracy=True)
            print 'saving model to %s...' % model_name
            self.model.save_weights(model_name)

    def add_dense(self):
        self.model.add(Dense(int(self.conf['--input_dim'])
                             , 1
                             , init=self.conf['--initialization']
                             , activation=self.conf['--activation']))

        # model.add(Activation(conf['--activation']))

        return self.model

    def add_lstm(self):
        self.model.add(LSTM(int(self.conf['--input_dim'])
                            , 1
                            , activation=self.conf['--activation']
                            , inner_activation=self.conf['--inner_activation']
                            , init=self.conf['--initialization']
                            , inner_init=self.conf['--inner_init']
                            , truncate_gradient=int(self.conf['--truncated_gradient'])
                            , return_sequences=False))

        #  model.add(Activation(conf['--activation']))
        return self.model

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