__author__ = 'jramapuram'

import data_manipulator
import pandas as pd
import numpy as np
from data_source import DataSource

from sklearn.cross_validation import train_test_split


class CSVReader(DataSource):
    def __init__(self, conf):
        self.conf = conf
        self.data = pd.DataFrame()
        self.path = conf['--input']
        self.test_ratio = float(conf['--test_split_ratio'])
        self.read_bad_lines = False

    def read_data(self):
        self.data = self.window_data(pd.read_csv(self.path, error_bad_lines=self.read_bad_lines).values)
        print 'read %s elements from %s' % (self.data.shape, self.path)
        print self.data
        return self.data

    def split_data(self):
        if self.data.empty:
            self.data = self.read_data()
        return train_test_split(self.data, test_size=self.test_ratio)

    def window_data(self, data):
        generator = data_manipulator.window(data, int(self.conf['--input_dim']))
        return np.array([item for item in generator])