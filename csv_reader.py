__author__ = 'jramapuram'

import data_manipulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_source import DataSource
from data_manipulator import plot_wave

from sklearn.cross_validation import train_test_split


class CSVReader(DataSource):
    def __init__(self, conf):
        self.conf = conf
        self.data = (np.matrix([]), np.matrix([]))
        self.input_column = self.conf['--input_col']
        self.use_cols = [self.input_column]
        self.has_classes = False

        if conf['--test_col'] is not None:
            self.test_column = self.conf['--test_col']
            self.classes = np.array
            self.use_cols.append(self.test_column)
            self.has_classes = True

        self.raw_data = pd.DataFrame()
        self.inputs = np.array
        self.path = conf['--input']
        self.test_ratio = float(conf['--test_ratio'])
        self.read_bad_lines = False

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def read_data(self):
        self.raw_data = pd.read_csv(self.path
                                    , error_bad_lines=self.read_bad_lines
                                    , usecols=self.use_cols)
        self.inputs = data_manipulator.normalize(self.raw_data[self.input_column].values.flatten()).T

        # TODO: Duplicate data by tiling + N(0,1) or something similar. Just tiling adds no new info [verified]
        # self.classes = np.tile(self.classes, 5)
        # self.inputs = np.tile(self.inputs, 5)

        if self.has_classes:
            self.classes = self.raw_data[self.test_column].values.flatten()
            # plot_wave(self.classes, 'original class data [pre window]')
            # self.data = (self.window_data(self.inputs), self.window_data(self.classes))
            self.data = (self.inputs, self.classes)
        else:
            # self.data = (self.window_data(self.inputs),)
            self.data = (self.inputs,)

        print 'Data Stats:\n\t-windowed %s elements (originally %s)\n\t-file:%s\n\t-columns: [%s]' \
              % (self.data[0].shape, self.inputs.shape, self.path, ', '.join(self.use_cols))

        if self.has_classes:
            return self.data[0], self.data[1]
        else:
            return self.data[0], None

    def get_classes(self):
        return self.classes

    def get_inputs(self):
        return self.inputs

    def read_raw_data(self):
        return self.raw_data

    def get_noise_count(self):
        if self.has_classes:
            return len(self.classes[np.where(self.classes > 0)])
        else:
            return -1

    def get_original_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def split_data(self):
        if self.data[0].size == 0:
            self.data = self.read_data()

        # TODO: Cleanup clunky
        if self.has_classes:
            # below does random shuffling, this might cause problems with the model learning useful things
            # (x_train, x_test), (y_train, y_test) = train_test_split(self.data[0], test_size=self.test_ratio)\
            #                                        , train_test_split(self.data[1], test_size=self.test_ratio)
            (x_train, x_test) = data_manipulator.split(self.data[0], self.test_ratio)
            (y_train, y_test) = data_manipulator.split(self.data[1], self.test_ratio)

            self.x_train = x_train.flatten()
            self.x_test = x_test.flatten()
            self.y_train = y_train.flatten()
            self.y_test = y_test.flatten()

            plot_wave(self.x_train, 'original train data')
            plot_wave(self.x_test, 'original test data')
            plot_wave(self.y_test, 'original test class')
            plot_wave(self.y_train, 'original train class')

            return (self.window_data(self.x_train), self.window_data(self.y_train))\
                , (self.window_data(self.x_test), self.window_data(self.y_test))
        else:
            # below does random shuffling, this might cause problems with the model learning useful things
            # (x_train, x_test) = train_test_split(self.data[0], test_size=self.test_ratio)
            (x_train, x_test) = data_manipulator.split(self.data[0], self.test_ratio)

            self.x_train = x_train.flatten()
            self.x_test = x_test.flatten()

            plot_wave(self.x_train, 'original train data')
            plot_wave(self.x_test, 'original test data')

            return (self.window_data(self.x_train), np.array([]))\
                , (self.window_data(self.x_test), np.array([]))

    def window_data(self, data):
        generator = data_manipulator.window(data, int(self.conf['--input_dim']))
        return np.array([item for item in generator])
