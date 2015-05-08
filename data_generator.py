__author__ = 'jramapuram'

import numpy as np

from random import randint
from math import sin, pi
from data_manipulator import window

def generate_sin_wave(input_size, num_waves, offset=-1):
    delta = 2*pi/(input_size-offset)  # for proper shifting
    one_wave = [sin(delta*i) for i in xrange(0, input_size)]
    return one_wave*num_waves

def add_amplitude_noise(conf, signal, num_errors=1):
    signal = np.array(signal)
    max_len = len(signal)
    for i in xrange(0, num_errors):
        location = randint(0, max_len - 1)
        signal[location] = signal[location] + np.random.normal(5, 0.1)  # TODO: parameterize this
        if 'noise_count' in conf:
            conf['noise_count'] += 1
        else:
            conf['noise_count'] = 1
    return signal

def generate_data(full_length, window_length, periods):
    wave = generate_sin_wave(full_length, periods)
    generator = window(wave, window_length)
    return np.array([item for item in generator])

def generate_test_data(conf):
    test_data = generate_data(int(conf['--input_dim'])*int(conf['--num_test_periods']), int(conf['--input_dim']), 1)
    return np.matrix([add_amplitude_noise(conf, row, 1).flatten() if randint(0, 13) == 1
                    else row.flatten() for row in test_data])

