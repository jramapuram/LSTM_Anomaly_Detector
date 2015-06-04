__author__ = 'jramapuram'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

from itertools import islice

def split_vector(vec, split_size):
    return np.reshape(vec[0:split_size*np.floor(len(vec)/float(split_size))]
                      , (-1, split_size))

def plot_wave(wave, title='wave function', minmax=[]):
    fig = plt.figure()
    plt.plot(wave)
    if not minmax:
        plt.ylim([np.min(wave), np.max(wave)])
    else:
        assert len(minmax) == 2
        plt.ylim(minmax)
    fig.suptitle(title)

# http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
def is_power2(num):
    # 'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)

def elementwise_square(list):
    return np.square(list)

def normalize(mat):
    return Normalizer().fit_transform(mat)

def split(mat, test_ratio):
    train_ratio = 1.0 - test_ratio
    train_index = np.floor(len(mat) * train_ratio)
    if mat.ndim == 2:
        return mat[0:train_index, :], mat[train_index + 1:len(mat) - 1, :]
    elif mat.ndim == 1:
        return mat[0:train_index], mat[train_index + 1:len(mat) - 1]
    else:
        print 'dimensionality is not 2 or 1!'
        raise NotImplementedError

# http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result