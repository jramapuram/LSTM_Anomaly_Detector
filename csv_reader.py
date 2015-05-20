__author__ = 'jramapuram'

import data_manipulator
import pandas as pd
import numpy as np
from data_source import DataSource
from data_manipulator import plot_wave

from sklearn.cross_validation import train_test_split


class CSVReader(DataSource):
    def __init__(self, conf):
        self.conf = conf
        self.data = (np.matrix([]), np.matrix([]))
        self.input_column = self.conf['--input_col']
        self.test_column = self.conf['--test_col']
        self.raw_data = pd.DataFrame()
        self.inputs = np.array
        self.classes = np.array
        self.path = conf['--input']
        self.test_ratio = float(conf['--test_ratio'])
        self.read_bad_lines = False

    def read_data(self):
        self.raw_data = pd.read_csv(self.path, error_bad_lines=self.read_bad_lines
                                    , usecols=[self.input_column, self.test_column])
        self.classes = self.raw_data[self.test_column].values.flatten()
        self.inputs = self.raw_data[self.input_column].values.flatten()

        # TODO: Duplicate data by tiling + N(0,1) or something similar. Just tiling adds no new info
        # self.classes = np.tile(self.classes, 5)
        # self.inputs = np.tile(self.inputs, 5)

        plot_wave(self.inputs, 'original input data [pre window]')
        plot_wave(self.classes, 'original class data [pre window]')

        self.data = (self.window_data(self.inputs), self.window_data(self.classes))

        print 'Data Stats:\n\t-windowed %s elements (originally %s)\n\t-file:%s\n\t-columns: [%s, %s]' \
              % (self.data[0].shape, self.inputs.shape, self.path, self.input_column, self.test_column)
        return data_manipulator.normalize(self.data[0]), self.data[1]

    def get_classes(self):
        return self.classes

    def get_inputs(self):
        return self.inputs

    def read_raw_data(self):
        return self.raw_data

    def get_noise_count(self):
        return len(self.classes[np.where(self.classes > 0)])

    def split_data(self):
        if self.data[0].size == 0:
            self.data = self.read_data()
        (x_train, x_test), (y_train, y_test) = train_test_split(self.data[0], test_size=self.test_ratio)\
                                               , train_test_split(self.data[1], test_size=self.test_ratio)
        return (x_train, y_train), (x_test, y_test)

    def window_data(self, data):
        generator = data_manipulator.window(data, int(self.conf['--input_dim']))
        return np.array([item for item in generator])