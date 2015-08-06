__author__ = 'jramapuram'

import os.path
import scipy
import numpy as np

from RTRL import RTRL, LSTM, SGD
from data_manipulator import elementwise_square
# from keras.models import Sequential
# from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, AutoEncoder, Activation
# from keras.layers.normalization import BatchNormalization
# from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from convolutional import Convolution1D, MaxPooling1D


class TimeDistributedAutoEncoder:
    def __init__(self, conf):
        self.conf = conf
        self.model_dir = ''
        self.model_name = ''
        self.encoder_sizes = []
        self.decoder_sizes = []
        self.models = []
        self.compiled = False

    @property
    def get_model_name(self):
        if not self.compiled:
            raise Exception("Cannot determine model name without it first being compiled");

        model_structure = 'weights_[%s]Enc_[%s]Dec_%dbatch_%s_autoencoder.dat'
        model_name = model_structure % ('_'.join(str(e) for e in self.encoder_sizes)
                                        , '_'.join(str(d) for d in self.decoder_sizes)
                                        , int(self.conf['--batch_size'])
                                        , self.conf['--model_type'])
        model_dir = model_name.replace("weights_", "").replace(".dat", "")
        from data_manipulator import create_dir
        create_dir(model_dir)
        return model_dir, model_name

    def compile(self, optimizer=None):
        for model in self.models:
            print model.get_config(verbose=1)
            if optimizer is not None:
                model.compile(loss=self.conf['--loss'], optimizer=optimizer)
            else:
                #model.compile(loss=self.conf['--loss'], optimizer=self.conf['--optimizer'])
                model.compile(loss=self.conf['--loss'], optimizer=SGD(rtrl=True))
        self.compiled = True

    def add_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        # self.models = [RTRL() for i in range(len(encoder_sizes))]
        self.models = [RTRL()]

        encoders = RTRL()
        decoders = RTRL()
        for i in range(0, len(encoder_sizes) - 1):
            encoders.add(Dense(encoder_sizes[i], encoder_sizes[i + 1]
                               , init=self.conf['--initialization']
                               , activation=self.conf['--activation']
                               , W_regularizer=l2()))

            decoders.add(Dense(decoder_sizes[i], decoder_sizes[i + 1]
                               , init=self.conf['--initialization']
                               , activation=self.conf['--activation']
                               , W_regularizer=l2()))

        self.models[0].add(AutoEncoder(encoder=encoders
                                       , decoder=decoders
                                       , output_reconstruction=True))
        return self.models

    # TODO: This doesnt work yet
    # (batch size, stack size, nb row, nb col)
    def add_conv_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        # self.models = [RTRL() for i in range(len(encoder_sizes))]
        self.models = [RTRL()]

        encoders = RTRL()
        decoders = RTRL()
        for i in range(0, len(encoder_sizes) - 1):
            encoders.add(Convolution1D(32, 3, 3
                                       , activation=self.conf['--activation']
                                       , init=self.conf['--initialization']
                                       , border_mode='valid'))
            encoders.add(Activation('relu'))
            encoders.add(MaxPooling1D())
            encoders.add(Convolution1D(32, 1, 1
                                       , activation=self.conf['--activation']
                                       , init=self.conf['--initialization']
                                       , border_mode='valid'))

            decoders.add(Convolution1D(32, 1, 1
                                       , activation=self.conf['--activation']
                                       , init=self.conf['--initialization']
                                       , border_mode='valid'))
            decoders.add(Activation('relu'))
            decoders.add(MaxPooling1D())

        self.models[0].add(AutoEncoder(encoder=encoders
                                       , decoder=decoders
                                       , output_reconstruction=(i == 0)))
        return self.models

    def add_lstm_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        # self.models = [RTRL() for i in range(len(encoder_sizes))]
        self.models = [RTRL()]

        encoders = RTRL()
        decoders = RTRL()

        for i in range(0, len(encoder_sizes) - 1):
            encoders.add(LSTM(encoder_sizes[i], encoder_sizes[i + 1]
                              , activation=self.conf['--activation']
                              , inner_activation=self.conf['--inner_activation']
                              , init=self.conf['--initialization']
                              , inner_init=self.conf['--inner_init']
                              , truncate_gradient=int(self.conf['--truncated_gradient'])
                              , return_sequences=True))
            #encoders.add(BatchNormalization(encoder_sizes[i + 1]))
            encoders.add(Dropout(0.5))
            decoders.add(LSTM(decoder_sizes[i], decoder_sizes[i + 1]
                              , activation=self.conf['--activation']
                              , inner_activation=self.conf['--inner_activation']
                              , init=self.conf['--initialization']
                              , inner_init=self.conf['--inner_init']
                              , truncate_gradient=int(int(self.conf['--truncated_gradient']))
                              , return_sequences=True))#not (i == len(encoder_sizes) - 1)))
    
        # self.models[0].add(AutoEncoder(encoder=encoders
        #                                , decoder=decoders
        #                                , output_reconstruction=True))
        self.models[0].add(encoders)
        self.models[0].add(decoders)

        return self.models

    def format_lstm_data(self, x):
        # Need to create a 3d vector [samples, timesteps, input_dim]
        if self.conf['--model_type'].strip().lower() == 'lstm':
            x = x[:, np.newaxis, :]
            print 'modified training data to fit LSTM: ', x.shape
        return x

    def unformat_lstm_data(self, x):
        # Need to create a 2d vector [samples, input_dim]
        if self.conf['--model_type'].strip().lower() == 'lstm':
            x = x[:, np.newaxis, :]
        return x

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        out = e_x / e_x.sum()
        return out

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def train_and_predict(self, x):
        if not self.compiled:
            self.compile()

        self.model_dir, self.model_name = self.get_model_name
        _ = self.load_model(os.path.join(self.model_dir, self.model_name), self.models[0])

        x = self.format_lstm_data(x)

        predictions = []
        for i in xrange(x.shape[0] - 1):
            if self.conf['--model_type'].strip().lower() == 'lstm':
                predictions.append(self.models[0].predict(x[i:i+1, :, :], batch_size=1, verbose=False))
                #current = self.models[0].predict(x[i:i+1, :, :], verbose=False)
                #future = self.models[0].predict(x[i+1:i+2, :, :], verbose=False)
                #ks_d, ks_p_value = scipy.stats.ks_2samp(current.flatten(), future.flatten())
                #predictions.append((ks_d, ks_p_value))
                # if ks_p_value < 0.05 and ks_d > 0.5:
                #     predictions.append(np.ones(1))
                # else:
                #     predictions.append(np.zeros(1))
                self.models[0].train_on_batch(x[i:i+1, :, :], x[i:i+1, :, :], accuracy=False)
            else:
                predictions.append(self.models[0].predict(x[i:i+1, :], verbose=False))
                self.models[0].train_on_batch(x[i:i+1, :], x[i:i+1, :], accuracy=True)

        predictions.append(predictions[-1])  # XXX
        predictions = np.array(predictions)
        print 'saving model to %s...' % os.path.join(self.model_dir, self.model_name)
        self.models[0].save_weights(os.path.join(self.model_dir, self.model_name), overwrite=True)

        if len(predictions.shape) == 4:
            predictions = np.squeeze(np.squeeze(predictions, (1,)), (1,))
        elif len(predictions.shape) == 3:
            predictions = np.squeeze(predictions, (1,))
        
        print 'predictions.shape: ', predictions.shape
        np.savetxt(os.path.join(self.model_dir, 'outputs.csv'), predictions, delimiter=',')
        return predictions

    def get_model(self):
        return self.models

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
