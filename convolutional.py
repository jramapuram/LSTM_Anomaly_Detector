__author__ = 'jramapuram'

from theano.tensor.signal import downsample
from keras.utils.theano_utils import shared_zeros
from keras.layers.core import Layer

import theano
import theano.tensor as T
import keras.initializations as initializations
import keras.activations as activations


class Convolution1D(Layer):
    def __init__(self, nb_filter, stack_size, filter_length,
                 init='glorot_uniform', activation='linear', weights=None,
                 image_shape=None, border_mode='valid', subsample_length=1):
        super(Convolution1D, self).__init__()

        nb_row = 1
        nb_col = filter_length
        subsample = (1,subsample_length)
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.image_shape = image_shape
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter,))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W
                                                  , border_mode=self.border_mode
                                                  , subsample=self.subsample
                                                  , image_shape=self.image_shape)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
                "nb_filter":self.nb_filter,
                "stack_size":self.stack_size,
                "init":self.init.__name__,
                "activation":self.activation.__name__,
                "image_shape":self.image_shape,
                "border_mode":self.border_mode,
                "subsample":self.subsample}


class MaxPooling1D(Layer):
    def __init__(self, pool_length=2, ignore_border=True):
        super(MaxPooling1D,self).__init__()

        poolsize = (1, pool_length)
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        self.params = []

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, self.poolsize, ignore_border=self.ignore_border)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
                "poolsize":self.poolsize,
                "ignore_border":self.ignore_border}