__author__ = 'jramapuram'

import os.path

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD


class AutoEncoder:
    def __init__(self, conf):
        self.conf = conf
        self.model = Sequential()

        # Ideally we need to pass in a 3d vector: (nb_samples, timesteps, input_dim)
        # This works for now. TODO: Explore alternative strategies or just get the timesteps?
        if self.conf['--model_type'].strip().lower() == 'lstm':
            self.model.add(Embedding(int(self.conf['--max_features']), int(self.conf['--input_dim'])))

    def train_autoencoder(self, X_train):
        self.model.get_config(verbose=1)
        if self.conf['--optimizer'] == 'sgd':
            # customize SGD as the defauly keras one does not use momentum or nesterov
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(loss=self.conf['--loss'], optimizer=sgd)
        else:
            self.model.compile(loss=self.conf['--loss'], optimizer=self.conf['--optimizer'])

        model_structure = 'weights_%din_%dhid_%dbatch_%depochs_%s_autoencoder.dat'
        model_name = model_structure % (int(self.conf['--input_dim'])
                                        , int(self.conf['--hidden_dim'])
                                        , int(self.conf['--batch_size'])
                                        , int(self.conf['--max_epochs'])
                                        , self.conf['--model_type'])
        model_exists = self.load_model(model_name, self.model)

        if not model_exists:
            print 'training new model using %s loss function & %s optimizer...' \
                  % (self.conf['--loss'], self.conf['--optimizer'])

            self.model.fit(X_train, X_train
                           , batch_size=int(self.conf['--batch_size'])
                           , nb_epoch=int(self.conf['--max_epochs'])
                           , validation_split=float(self.conf['--validation_ratio'])
                           , show_accuracy=True)
            print 'saving model to %s...' % model_name
            self.model.save_weights(model_name)

    def add_autoencoder(self, input_size, hidden_size, output_size):
        self.model.add(Dense(input_size, hidden_size
                             , init=self.conf['--initialization']
                             , activation=self.conf['--activation']))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(hidden_size, output_size
                             , init=self.conf['--initialization']
                             , activation=self.conf['--activation']))
        # self.model.add(Activation('relu'))
        # self.model.add(Activation(self.conf['--activation']))
        return self.model

    def add_lstm_autoencoder(self, input_size, hidden_size, output_size, is_final_layer=False):
        self.model.add(LSTM(input_size, hidden_size
                            , activation=self.conf['--activation']
                            , inner_activation=self.conf['--inner_activation']
                            , init=self.conf['--initialization']
                            , inner_init=self.conf['--inner_init']
                            , truncate_gradient=int(self.conf['--truncated_gradient'])
                            , return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(hidden_size, output_size
                       , activation=self.conf['--activation']
                       , inner_activation=self.conf['--inner_activation']
                       , init=self.conf['--initialization']
                       , inner_init=self.conf['--inner_init']
                       , truncate_gradient=int(int(self.conf['--truncated_gradient']))
                       , return_sequences=(not is_final_layer)))
        # self.model.add(Activation(self.conf['--activation']))
        return self.model

    def predict_mse_mean(self, test):
        import numpy as np
        mse_predictions = np.array([np.mean(item) for item in self.predict_mse(test)]).flatten()
        assert len(mse_predictions) == len(test)
        return mse_predictions

    def predict_mse(self, test):
        import numpy as np
        from data_manipulator import elementwise_square
        mse_predictions = np.array([(elementwise_square((xtrue - xpred).T)).flatten()
                                    for xtrue, xpred in zip(test, self.predict(test))])
        return mse_predictions

    def predict(self, test):
        return self.model.predict(test, verbose=False)

    def get_model(self):
        return self.model

    def get_model_type(self):
        return self.conf['--model_type']

    @staticmethod
    def load_model(path_str, model):
        if os.path.isfile(path_str):
            print 'model found, loading existing model...'
            model.load_weights(path_str)
            return True
        else:
            print 'model does not exist...'
            return False
