"""LSTM Anomaly Detector.

Usage:
    lstm_anomaly_detector.py (-h | --help | --version)
    lstm_anomaly_detector.py synthetic [--quiet] [--plot_live] [--model_type=<type>] [--inner_activation=<activationFn>] [--num_periods=<periods>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--inner_init=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs_classifier=<iter>] [--truncated_gradient=<bool>] [--test_ratio=<ratio>] [--validation_ratio=<ratio>]
    lstm_anomaly_detector.py csv (--input=<FILE>) (--input_col=<column>) [--test_col=<column>] [--quiet] [--plot_live] [--model_type=<type>] [--inner_activation=<activationFn>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--inner_init=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs_classifier=<iter>] [--truncated_gradient=<bool>] [--test_ratio=<ratio>] [--validation_ratio=<ratio>]

Options:
    -h --help                           show this
    --version                           show version
    --quiet                             print less text [default: False]
    --plot_live                         if false the images are stored in a directory instead of shown [default: False]
    --input=<FILE>                      csv file if in csv mode
    --input_col=<column>                the column to read our value data from [default: value]
    --test_col=<column>                 the column which contains the binary vector for the anomaly
    --model_type=<type>                 either "lstm" or "classical" [default: lstm]
    --num_periods=<periods>             number of periods of the signal to generate [default: 32]
    --activation=<activationFn>         activation function [default: sigmoid]
    --inner_activation=<activationFn>   inner activation used for lstm [default: hard_sigmoid]
    --input_dim=<num>                   number of values into the input layer [default: 64]
    --hidden_dim=<num>                  number of values for the hidden layer [default: 32]
    --batch_size=<num>                  number of values to batch for performance [default: 64]
    --initialization=<type>             type of weight initialization [default: glorot_uniform]
    --inner_init=<type>                 inner activation for LSTM [default: orthogonal]
    --optimizer=<type>                  optimizer type [default: adam]
    --loss=<lossFn>                     the lo ss function [default: mean_squared_error]
    --max_epochs_classifier=<iter>      the max number of epochs to iterate for the classifier [default: 1000]
    --truncated_gradient=<bool>         1 or -1 for truncation of gradient [default: -1]
    --test_ratio=<ratio>                number between 0 and 1 for which the data is split for test [default: 0.1]
    --validation_ratio=<ratio>          number between 0 and 1 for which the data is split for validation [default: 0.3]

"""

__author__ = 'jramapuram'

import numpy as np
from data_manipulator import is_power2, Plot, roll_rows, elementwise_square
from docopt import docopt
from autoencoder import TimeDistributedAutoEncoder
from csv_reader import CSVReader
from data_generator import DataGenerator

if __name__ == "__main__":
    conf = docopt(__doc__, version='LSTM Anomaly Detector 0.1')
    ae = TimeDistributedAutoEncoder(conf)
    p = Plot(conf, ae)

    # determine whether to pull in fake data or a csv file
    if conf['synthetic']:
        print 'generating synthetic data....'
        source = DataGenerator(conf, p)
    else:
        print 'reading from csv file...'
        source = CSVReader(conf, p)

    # pull in the data
    (x_train, y_train), (x_test, y_test) = source.split_data()
    print 'X_train.shape %s | Y_train.shape: %s' % (x_train.shape, y_train.shape)
    print 'X_test.shape %s  | Y_test.shape: %s' % (x_test.shape, y_test.shape)

    # build an LSTM or a regular autoencoder
    print 'building %s autoencoder...' % conf['--model_type']

    # Add the required layers
    model_type = conf['--model_type'].strip().lower()
    if model_type == 'lstm':
        # Deep autoencoder
        ae.add_lstm_autoencoder([int(conf['--input_dim']), int(conf['--hidden_dim'])
                                , int(conf['--hidden_dim'])/2, int(conf['--hidden_dim']) / 4]
                                , [int(conf['--hidden_dim'])/4, int(conf['--hidden_dim']) / 2
                                , int(conf['--hidden_dim']), int(conf['--input_dim'])])
        # Single autoencoder
        # ae.add_lstm_autoencoder([int(conf['--input_dim']), int(conf['--hidden_dim'])]
        #                         , [int(conf['--hidden_dim']), int(conf['--input_dim'])])

    elif model_type == 'conv':
        ae.add_conv_autoencoder([int(conf['--input_dim']), int(conf['--hidden_dim'])]
                                , [int(conf['--hidden_dim']), int(conf['--input_dim'])])
    else:
        # Deep autoencoder:
        ae.add_autoencoder([int(conf['--input_dim']), int(conf['--hidden_dim'])
                           , int(conf['--hidden_dim'])/2, int(conf['--hidden_dim']) / 4]
                           , [int(conf['--hidden_dim'])/4, int(conf['--hidden_dim']) / 2
                           , int(conf['--hidden_dim']), int(conf['--input_dim'])])

        # Single autoencoder:
        # ae.add_autoencoder([int(conf['--input_dim']), int(conf['--hidden_dim'])],
        #                    [int(conf['--hidden_dim']), int(conf['--input_dim'])])

    pred = ae.train_and_predict(x_train)
    print 'train original shape: %s, train predictions shape: %s' % x_train.shape, pred.shape
    p.plot_wave(pred, 'train predictions')
    p.plot_wave(ae.sigmoid((elementwise_square(x_train - pred)) / x_train.shape[0]), 'train mse')

    if conf['--test_col'] is not None:
        # run data through autoencoder (so that it can be pulled into classifier)
        ae_predictions = ae.predict_mse(x_train)
        print 'ae_predictions.shape: ', ae_predictions.shape

        # format the output vectors
        y_train_vector = np.array([1 if y_train[np.where(item >= 1)].size else 0 for item in y_train])
        y_test_vector = np.array([1 if y_test[np.where(item >= 1)].size else 0 for item in y_test])
        print 'y_train_vector shape :', y_train_vector.shape

        from classifier import Classifier
        cf = Classifier('classical', conf)
        print 'building %s classifier...' % cf.get_model_type()
        cf.add_dense()
        cf.train_classifier(ae_predictions, y_train_vector)
        cf_model = cf.get_model()

        # predict nom nom
        predictions = np.array([1 if item >= 0.5 else 0
                                for item in cf_model.predict(ae.predict_mse(x_test))])
        predictions_train = np.array([1 if item >= 0.5 else 0
                                      for item in cf_model.predict(ae.predict_mse(x_train))])
        p.plot_wave(predictions, '[test] Classifier Predictions')
        p.plot_wave(predictions_train, '[train] Classifier predicitions')

        print 'classifier prediction size: ', predictions.shape
        print '[test] number of anomalies detected: ', len(predictions[np.where(predictions > 0)])
        print '[train] number of anomalies detected: ', len(predictions_train[np.where(predictions_train > 0)])

    print 'number of anomalies originally: ', source.get_noise_count()
    p.show()
