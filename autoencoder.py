__author__ = 'jramapuram'

import math
import os.path
import numpy as np
import numpy as np
import six

import chainer
import numpy as np
import chainer.functions as F

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from data_manipulator import elementwise_square


class TimeDistributedAutoEncoder:
    def __init__(self, conf):
        cuda.init(0)
        self.conf = conf
        self.model_dir = ''
        self.model_name = ''
        self.encoder_sizes = []
        self.decoder_sizes = []
        self.encoders = {}
        self.decoders = {}
        self.model = None
        self.state = None
        self.optimizer = optimizers.Adam()  # Parametrize

    @property
    def get_model_name(self):
        model_structure = 'weights_[%s]Enc_[%s]Dec_%dbatch_%s_autoencoder.dat'
        model_name = model_structure % ('_'.join(str(e) for e in self.encoder_sizes)
                                        , '_'.join(str(d) for d in self.decoder_sizes)
                                        , int(self.conf['--batch_size'])
                                        , self.conf['--model_type'])
        model_dir = model_name.replace("weights_", "").replace(".dat", "")
        from data_manipulator import create_dir
        create_dir(model_dir)
        return model_dir, model_name

    def make_initial_state(self, batch_size, train=True):
        state = {}
        print 'enc sizes: ', self.encoder_sizes
        print 'dec sizes: ', self.decoder_sizes
        for i in range(0, len(self.encoder_sizes)):
            state['c_e' + str(i)] = chainer.Variable(cuda.zeros((batch_size, self.encoder_sizes[i]),
                                                                dtype=np.float32),
                                                     volatile=not train)
            state['h_e' + str(i)] = chainer.Variable(cuda.zeros((batch_size, self.encoder_sizes[i]),
                                                                dtype=np.float32),
                                                     volatile=not train)
        for i in range(0, len(self.decoder_sizes)):
            state['c_d' + str(i)] = chainer.Variable(cuda.zeros((batch_size, self.decoder_sizes[i]),
                                                                dtype=np.float32),
                                                     volatile=not train)
            state['h_d' + str(i)] = chainer.Variable(cuda.zeros((batch_size, self.decoder_sizes[i]),
                                                                dtype=np.float32),
                                                     volatile=not train)
        return state

    def forward_one_step_lstm(self, x_data, y_data, state, train=True):
        x_data = cuda.to_gpu(x_data)
        y_data = cuda.to_gpu(y_data)
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        # Encoder section : Special case for 0 as it uses raw data x
        state['h_e_in0'] = self.encoders['le_x0'](F.dropout(x, ratio=0.0, train=train)) + self.encoders['le_h0'](state['h_e0'])
        for i in range(0, len(self.encoder_sizes) - 1):
            num = str(i)
            num_plus_one = str(i + 1)
            state['c_e' + num], state['h_e' + num] = F.lstm(state['c_e' + num], state['h_e_in' + num])
            state['h_e_in' + num_plus_one] = self.encoders['le_x' + num_plus_one](F.dropout(state['h_e' + num], train=train)) \
                                             + self.encoders['le_h' + num_plus_one](state['h_e' + num_plus_one])

        # Decoder section: Special case for 0 as it uses last encoder state
        state['c_d0'], state['h_d0'] = F.lstm(state['c_d0'], state['h_e_in' + str(i + 1)])
        state['h_d_in0'] = self.decoders['ld_x0'](F.dropout(state['h_d0'], train=train)) \
                                + self.decoders['ld_h0'](state['h_d0'])
        i = len(self.decoder_sizes) - 1
        for i in range(1, len(self.decoder_sizes) - 1):
            num = str(i)
            num_minus_one = str(i - 1)
            state['h_d_in' + num] = self.decoders['ld_x' + num](F.dropout(state['h_d' + num_minus_one], train=train)) \
                                    + self.decoders['ld_h' + num](state['h_d' + num])
            state['c_d' + num], state['h_d' + num] = F.lstm(state['c_d' + num], state['h_d_in' + num])

        y = self.decoders['ld_h' + str(i)](F.dropout(state['h_d' + str(i)], train=train))

        print 'y=', y.__len__()
        print 't=', t.__len__()
        return state, F.mean_squared_error(y, t)

    def add_autoencoder(self, encoder_sizes, decoder_sizes):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes

        for i in range(0, len(encoder_sizes) - 1):
            self.encoders['e' + str(i)] = F.Linear(encoder_sizes[i], encoder_sizes[i + 1])
            self.decoders['d' + str(i)] = F.Linear(decoder_sizes[i], decoder_sizes[i + 1])
        layers = self.encoders.copy()
        self.model = FunctionSet(layers.update(self.decoders))
        for param in self.model.parameters:
            param[:] = np.random.uniform(-0.1, 0.1, param.shape)
        self.model.to_gpu()

        return self.model

    def add_lstm_autoencoder(self, encoder_sizes=[], decoder_sizes=[]):
        assert(len(encoder_sizes) != 0 and len(decoder_sizes) != 0)
        assert(len(encoder_sizes) == len(decoder_sizes))

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes

        # Initialize the first layers outsize
        self.encoders['le_x0'] = F.Linear(encoder_sizes[0], 4 * encoder_sizes[0])
        self.encoders['le_h0'] = F.Linear(encoder_sizes[0], 4 * encoder_sizes[0])
        self.decoders['ld_x0'] = F.Linear(decoder_sizes[0], 4 * decoder_sizes[0])
        self.decoders['ld_h0'] = F.Linear(decoder_sizes[0], 4 * decoder_sizes[0])
        
        for i in range(1, len(encoder_sizes)):
            print 'creating encoder layer: ', [encoder_sizes[i - 1], 4 * encoder_sizes[i]]
            self.encoders['le_x' + str(i)] = F.Linear(encoder_sizes[i - 1], 4 * encoder_sizes[i])
            self.encoders['le_h' + str(i)] = F.Linear(encoder_sizes[i], 4 * encoder_sizes[i])

        for i in range(1, len(decoder_sizes)):
            self.decoders['ld_x' + str(i)] = F.Linear(decoder_sizes[i - 1], 4 * decoder_sizes[i])
            if i == len(decoder_sizes) - 1:
                self.decoders['ld_h' + str(i)] = F.Linear(decoder_sizes[i], decoder_sizes[i])
            else:
                self.decoders['ld_h' + str(i)] = F.Linear(decoder_sizes[i], 4 * decoder_sizes[i])

        layers = self.encoders.copy()
        layers.update(self.decoders)

        self.model = FunctionSet(**layers)
        for param in self.model.parameters:
            param[:] = np.random.uniform(-0.1, 0.1, param.shape)
        cuda.init()
        self.model.to_gpu()
        self.state = self.make_initial_state(int(self.conf['--batch_size']), train=True)
        return self.model

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

    def evaluate_lstm(self, dataset):
        sum_log_perp = cuda.zeros(())
        state = self.make_initial_state(batch_size=1, train=False)
        for i in six.moves.range(dataset.size - 1):
            x_batch = dataset[i:i + 1]
            y_batch = dataset[i + 1:i + 2]
            state, loss = self.forward_one_step_lstm(x_batch, y_batch, state, train=False)
            sum_log_perp += loss.data.reshape(())

        return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))

    def train_and_predict(self, x):
        self.model_dir, self.model_name = self.get_model_name
        #_ = self.load_model(os.path.join(self.model_dir, self.model_name), self.models[0])
        # x = self.format_lstm_data(x)
        x = x.astype(np.float32)
        predictions = []
        self.state, loss = self.forward_one_step_lstm(x[0:1, :], x[0:1, :], self.state, train=True)
        for i in xrange(x.shape[0] - 1):
            if self.conf['--model_type'].strip().lower() == 'lstm':
                current_data = np.array([x[i:i+1, :], x[i:i+1, :]])
                predictions.append(self.evaluate_lstm(current_data))
                self.state, loss = self.forward_one_step_lstm(x[i:i+1, :], x[i:i+1, :], self.state, train=True)
            else:
                predictions.append(self.models[0].predict(x[i:i+1, :], verbose=False))
                self.models.train_on_batch(x[i:i+1, :], x[i:i+1, :], accuracy=True)

        predictions.append(predictions[-1])  # XXX
        print 'saving model to %s...' % os.path.join(self.model_dir, self.model_name)
        #self.models[0].save_weights(os.path.join(self.model_dir, self.model_name), overwrite=True)

        #predictions = np.squeeze(np.squeeze(np.array(predictions), (1,)), (1,))
        predictions = np.array(predictions)
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
