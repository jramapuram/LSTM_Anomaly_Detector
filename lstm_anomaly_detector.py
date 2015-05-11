"""LSTM Anomaly Detector.

Usage:
    lstm_anomaly_detector.py (-h | --help | --version)
    lstm_anomaly_detector.py synthetic [--quiet] [--model_type=<type>] [--inner_init=<activationFn>] [--num_periods=<periods>] [--num_test_periods=<periods>] [--max_features=<features>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs=<iter>] [--truncated_gradient=<bool>]
    lstm_anomaly_detector.py csv (--input=<FILE>) [--quiet] [--model_type=<type>] [--inner_init=<activationFn>] [--max_features=<features>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs=<iter>] [--truncated_gradient=<bool>] [--test_split_ratio=<ratio>]

Options:
    -h --help                     show this
    --version                     show version
    --quiet                       print less text [default: False]
    --input=<FILE>                csv file if in csv mode
    --model_type=<type>           either "lstm" or "autoencoder" [default: lstm]
    --num_periods=<periods>       number of periods of the signal to generate [default: 16]
    --num_test_periods=<periods>  number of periods of the signal to generate [default: 4]
    --max_features=<features>     number of max features used for LSTM embeddings [default: 20000]
    --activation=<activationFn>   activation function [default: tanh]
    --inner_init=<activationFn>   inner activation used for lstm [default: tanh]
    --input_dim=<num>             number of values into the input layer [default: 128]
    --hidden_dim=<num>             number of values for the hidden layer [default: 64]
    --batch_size=<num>            number of values to batch for performance [default: 64]
    --initialization=<type>       type of weight initialization [default: glorot_normal]
    --optimizer=<type>            optimizer type [default: adam]
    --loss=<lossFn>               the loss function [default: mean_squared_error]
    --max_epochs=<iter>           the max number of epochs to iterate for [default: 10]
    --truncated_gradient=<bool>   1 or -1 for truncation of gradient [default: -1]
    --test_split_ratio=<ratio>    number between 0 and 1 for which the test is split into [default: 0.1]

"""

__author__ = 'jramapuram'

import numpy as np
import matplotlib.pyplot as plt
import data_generator
import data_manipulator
from docopt import docopt
from csv_reader import CSVReader



if __name__ == "__main__":
    conf = docopt(__doc__, version='LSTM Anomaly Detector 0.1')

    # determine whether to pull in fake data or a csv file
    if conf['synthetic']:
        print 'generating synthetic data....'
        X_train = data_generator.generate_data(int(conf['--input_dim']), int(conf['--input_dim']), int(conf['--num_periods']))
        print 'X_train size: %s | input_size: %d' % (X_train.shape, int(conf['--input_dim']))
        X_test = data_generator.generate_test_data(conf)
        print 'X_test size: %s | number of noise samples added: %d' % (X_test.shape, conf['noise_count'])
    else:
        reader = CSVReader(conf)
        X_train, X_test = reader.split_data()
        print 'X_train.shape %s | X_test.shape: %s' % (X_train.shape, X_test.shape)

    # for item in X_train:
    #     print 'train shp: ', item.shape
    #
    # for item in X_test:
    #     print 'test shp: ', item.shape

    # build an LSTM or a regular autoencoder
    from autoencoder import AutoEncoder
    print 'building %s autoencoder...' % conf['--model_type']
    ae = AutoEncoder(conf, X_train, None)
    model = ae.get_model()

    # evaluate the model
    predictions = [model.predict_proba(item, batch_size=1, verbose=False) for item in X_test]
    mse_prediction = [np.array(data_manipulator.elementwise_square((xtrue - xpred).T)).flatten() for xtrue, xpred in zip(X_test, predictions)]
    mse_prediction = np.array(mse_prediction).flatten()

    print 'plotting results of anomaly detection...'
    data_manipulator.plot_wave(mse_prediction, 'mse prediction on sliding window')
    data_manipulator.plot_wave(np.ravel(X_test), 'test wave unrolled')
    X_test_mean = np.array([np.mean(row) for row in X_test])
    data_manipulator.plot_wave(np.ravel(X_test_mean), 'test wave mean approx')

    # XXX: Fix with proper classical tests like grubbs, etc.
    print 'anomaly detector caught ~ %d anomalies' % len(mse_prediction[np.where(mse_prediction > 5)])
    plt.show()