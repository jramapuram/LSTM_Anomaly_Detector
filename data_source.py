__author__ = 'jramapuram'

from abc import ABCMeta, abstractmethod


class DataSource(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def split_data(self):
        pass

    @abstractmethod
    def get_noise_count(self):
        pass