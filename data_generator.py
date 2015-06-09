__author__ = 'jramapuram'

import numpy as np

from data_source import DataSource
from random import randint
from math import sin, pi
from data_manipulator import window, split, normalize
# from sklearn.cross_validation import train_test_split

class DataGenerator(DataSource):
    def __init__(self, conf, plotter):
        self.conf = conf
        self.p = plotter
        self.data = np.matrix([])
        self.x_train = np.matrix([])
        self.x_test = np.matrix([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.noise_count = 0

    @staticmethod
    def generate_sin_wave(input_size, num_waves, offset=1):
        delta = 2 * pi / (input_size - offset)  # for proper shifting
        one_wave = [sin(delta * i) for i in xrange(0, input_size)]
        return normalize(np.array(one_wave * num_waves)).flatten()
        # return one_wave * num_waves

    def add_amplitude_noise(self, signal, num_errors=1):
        # signal = np.array(signal)
        max_len = len(signal)
        for i in xrange(0, num_errors):
            location = randint(0, max_len - 1)
            signal[location] = signal[location] + np.random.normal(5, 0.1)  # TODO: parameterize this
            self.noise_count += 1
        return signal

    def read_data(self):
        wave = self.generate_sin_wave(int(self.conf['--input_dim'])
                                      , int(self.conf['--num_periods']))
        print len(wave)
        self.p.plot_wave(wave, 'train')
        generator = window(wave, int(self.conf['--input_dim']))
        self.data = np.array([item for item in generator])
        self.x_train, self.x_test = split(self.data, float(self.conf['--test_ratio']))
        self.x_test = self.add_amplitude_noise(self.x_test, 13)  # XXX

        print self.x_train.shape, self.x_test.shape
        return self.x_train

    def split_data(self):
        if self.data.size == 0:
            self.read_data()
        # TODO: Generate a y output vector where noise is added
        return (self.x_train, np.array([])), (self.x_test, np.array([]))

    def get_noise_count(self):
        return self.noise_count
