__author__ = 'jramapuram'

import os.path

from numpy import roll, newaxis
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, AutoEncoder
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils.dot_utils import Grapher
from keras.regularizers import l2


class TimeDistributedAutoEncoder:
    def __init__(self, conf):
        self.conf = conf
        self.model = Sequential()

    def train_autoencoder(self, X_train, rotate_forward_count=-1):
        self.model.get_config(verbose=1)
        if self.conf['--optimizer'] == 'sgd':
            # customize SGD as the default keras constructor does not use momentum or nesterov
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

        Grapher().plot(self.model, 'model.png')

        if not model_exists:
            print 'training new model using %s loss function & %s optimizer...' \
                  % (self.conf['--loss'], self.conf['--optimizer'])

            # Need to create a 3d vector [samples, timesteps, input_dim]
            if self.conf['--model_type'].strip().lower() == 'lstm':
                X_train = X_train[:, newaxis, :]
                print 'modified training data to fit LSTM: ', X_train.shape
            self.model.fit(X_train, roll(X_train, rotate_forward_count, axis=0)
                           , batch_size=int(self.conf['--batch_size'])
                           , nb_epoch=int(self.conf['--max_epochs'])
                           #, validation_split=float(self.conf['--validation_ratio'])
                           , show_accuracy=True
                           , shuffle=False)
            print 'saving model to %s...' % model_name
            self.model.save_weights(model_name)

    def add_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        encoders = []
        decoders = []
        for i in range(0, len(encoder_sizes) - 1):
            encoders.append(Dense(encoder_sizes[i], encoder_sizes[i + 1]
                                  , init=self.conf['--initialization']
                                  , activation=self.conf['--activation']
                                  , W_regularizer=l2()))
            decoders.append(Dense(decoder_sizes[i], decoder_sizes[i + 1]
                                  , init=self.conf['--initialization']
                                  , activation=self.conf['--activation']
                                  , W_regularizer=l2()))

        self.model.add(AutoEncoder(encoders=encoders
                                   , decoders=decoders
                                   , tie_weights=True
                                   , output_reconstruction=True))
        return self.model

    def add_lstm_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        encoders = []
        decoders = []
        for i in range(0, len(encoder_sizes) - 1):
            encoders.append(LSTM(encoder_sizes[i], encoder_sizes[i + 1]
                                 , activation=self.conf['--activation']
                                 , inner_activation=self.conf['--inner_activation']
                                 , init=self.conf['--initialization']
                                 , inner_init=self.conf['--inner_init']
                                 , truncate_gradient=int(self.conf['--truncated_gradient'])
                                 , return_sequences=True))
            decoders.append(LSTM(decoder_sizes[i], decoder_sizes[i + 1]
                                 , activation=self.conf['--activation']
                                 , inner_activation=self.conf['--inner_activation']
                                 , init=self.conf['--initialization']
                                 , inner_init=self.conf['--inner_init']
                                 , truncate_gradient=int(int(self.conf['--truncated_gradient']))
                                 , return_sequences=not (i == len(encoder_sizes) - 1)))

        self.model.add(AutoEncoder(encoders=encoders
                                   , decoders=decoders
                                   , tie_weights=True
                                   , output_reconstruction=True))
        return self.model

    def predict_mse_mean(self, test):
        predictions = self.predict_mse(test) #This looks reasonable
        # predictions = self.predict(test)
        mse_predictions = predictions.mean(axis=1)
        assert len(mse_predictions) == len(test)
        return mse_predictions

    def predict_mse(self, test):
        from data_manipulator import elementwise_square
        predictions = self.predict(test)
        mse_predictions = elementwise_square(Normalizer().fit_transform(test)
                                             - Normalizer().fit_transform(predictions))
        return mse_predictions

    def predict(self, test):
        # Need to create a 3d vector [samples, timesteps, input_dim]
        if self.conf['--model_type'].strip().lower() == 'lstm':
            test = test[:, newaxis, :]
        predictions = self.model.predict(test, int(self.conf['--batch_size']), verbose=0)
        if len(predictions.shape) > 2:  # Resize the LSTM outputs
            predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
        return predictions

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
