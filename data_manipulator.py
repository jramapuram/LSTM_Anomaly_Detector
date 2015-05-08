__author__ = 'jramapuram'

import numpy as np
import matplotlib.pyplot as plt

from itertools import islice

def split_vector(vec, split_size):
    return np.reshape(vec, (-1, split_size))

def plot_wave(wave, title='wave function'):
    fig = plt.figure()
    plt.plot(wave)
    fig.suptitle(title)

def elementwise_square(list):
    return [i ** 2 for i in list]

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