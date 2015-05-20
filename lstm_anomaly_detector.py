"""LSTM Anomaly Detector.

Usage:
    lstm_anomaly_detector.py (-h | --help | --version)
    lstm_anomaly_detector.py synthetic [--quiet] [--model_type=<type>] [--inner_activation=<activationFn>] [--num_periods=<periods>] [--max_features=<features>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--inner_init=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs=<iter>] [--truncated_gradient=<bool>] [--test_ratio=<ratio>] [--validation_ratio=<ratio>]
    lstm_anomaly_detector.py csv (--input=<FILE>) (--input_col=<column>) (--test_col=<column>) [--quiet] [--model_type=<type>] [--inner_activation=<activationFn>] [--max_features=<features>] [--activation=<activationFn>] [--input_dim=<num>] [--hidden_dim=<num>] [--batch_size=<num>] [--initialization=<type>] [--inner_init=<type>] [--optimizer=<type>] [--loss=<lossFn>] [--max_epochs=<iter>] [--truncated_gradient=<bool>] [--test_ratio=<ratio>] [--validation_ratio=<ratio>]

Options:
    -h --help                           show this
    --version                           show version
    --quiet                             print less text [default: False]
    --input=<FILE>                      csv file if in csv mode
    --input_col=<column>                the column to read our value data from [default: value]
    --test_col=<column>                 the column which contains the binary vector for the anomaly [default: is_anomaly]
    --model_type=<type>                 either "lstm" or "classical" [default: lstm]
    --num_periods=<periods>             number of periods of the signal to generate [default: 32]
    --max_features=<features>           number of max features used for LSTM embeddings [default: 20000]
    --activation=<activationFn>         activation function [default: sigmoid]
    --inner_activation=<activationFn>   inner activation used for lstm [default: hard_sigmoid]
    --input_dim=<num>                   number of values into the input layer [default: 64]
    --hidden_dim=<num>                  number of values for the hidden layer [default: 64]
    --batch_size=<num>                  number of values to batch for performance [default: 64]
    --initialization=<type>             type of weight initialization [default: glorot_uniform]
    --inner_init=<type>                 inner activation for LSTM [default: orthogonal]
    --optimizer=<type>                  optimizer type [default: adam]
    --loss=<lossFn>                     the loss function [default: mean_squared_error]
    --max_epochs=<iter>                 the max number of epochs to iterate for [default: 1000]
    --truncated_gradient=<bool>         1 or -1 for truncation of gradient [default: -1]
    --test_ratio=<ratio>                number between 0 and 1 for which the data is split for test [default: 0.1]
    --validation_ratio=<ratio>          number between 0 and 1 for which the data is split for validation [default: 0.3]

"""

__author__ = 'jramapuram'

import numpy as np
import matplotlib.pyplot as plt
from data_manipulator import is_power2, plot_wave
from docopt import docopt
from csv_reader import CSVReader
from data_generator import DataGenerator

if __name__ == "__main__":
    conf = docopt(__doc__, version='LSTM Anomaly Detector 0.1')
    # GPU needs power of 2
    assert is_power2(int(conf['--input_dim']))
    assert is_power2(int(conf['--hidden_dim']))

    # determine whether to pull in fake data or a csv file
    if conf['synthetic']:
        print 'generating synthetic data....'
        source = DataGenerator(conf)
    else:
        print 'reading from csv file...'
        source = CSVReader(conf)

    # pull in the data
    (x_train, y_train), (x_test, y_test) = source.split_data()
    print 'X_train.shape %s | Y_train.shape: %s' % (x_train.shape, y_train.shape)
    print 'X_test.shape %s  | Y_test.shape: %s' % (x_test.shape, y_test.shape)

    # build an LSTM or a regular autoencoder
    from autoencoder import AutoEncoder
    print 'building %s autoencoder...' % conf['--model_type']
    ae = AutoEncoder(conf)

    # Add the required layers
    print conf['--model_type'].strip().lower()
    if conf['--model_type'].strip().lower() == 'lstm':
        ae.add_lstm_autoencoder(int(conf['--input_dim'])
                                , int(conf['--hidden_dim'])
                                , int(conf['--hidden_dim']) / 2)
        ae.add_lstm_autoencoder(int(conf['--hidden_dim']) / 2
                                , int(conf['--hidden_dim'])
                                , int(conf['--input_dim'])
                                , is_final_layer=True)
    else:
        ae.add_autoencoder(int(conf['--input_dim'])
                           , int(conf['--hidden_dim'])
                           , int(conf['--hidden_dim']) / 2)
        ae.add_autoencoder(int(conf['--hidden_dim']) / 2
                           , int(conf['--hidden_dim'])
                           , int(conf['--input_dim']))

    ae.train_autoencoder(x_train)
    model = ae.get_model()

    # run data through autoencoder (so that it can be pulled into classifier)
    ae_predictions = ae.predict_mse(x_train)
    ae_mean_predictions = ae.predict_mse_mean(x_train)
    plot_wave(ae_mean_predictions, 'training autoencoder mse')
    print 'ae_pred shape : %s | ae_mean_predictions shape: %s' % (ae_predictions.shape, ae_mean_predictions.shape)
    # np.savetxt("training_mse.csv", ae_mean_predictions, delimiter=",")

    # predict on just the test data
    ae_test_predictions = ae.predict(x_test)
    ae_test_mean_predictions = ae.predict_mse_mean(x_test)
    plot_wave(ae_test_mean_predictions, 'test autoencoder mse')
    print 'ae_test_pred shape : %s | ae_test_mean_predictions shape: %s' \
          % (ae_test_predictions.shape, ae_test_mean_predictions.shape)

    # A little hacky, make this globally applicable
    if not conf['synthetic']:
        # format the output vectors
        y_train_vector = np.array([1 if y_train[np.where(item >= 1)].size else 0 for item in y_train])
        y_test_vector = np.array([1 if y_test[np.where(item >= 1)].size else 0 for item in y_test])
        print 'y_train_vector shape :', y_train_vector.shape

        # build a classifier
        from classifier import Classifier
        cf = Classifier('classical', conf)
        print 'building classifier...'
        # cf.add_lstm() # TODO: Find out why this breaks stuff
        cf.add_dense()
        cf.train_classifier(ae_predictions, y_train_vector)
        cf_model = cf.get_model()

        # predict nom nom
        predictions = np.array([1 if item >= 0.5 else 0
                                for item in cf_model.predict(ae.predict_mse(x_test))])
        predictions_train = np.array([1 if item >= 0.5 else 0
                                      for item in cf_model.predict(ae.predict_mse(x_train))])
        plot_wave(predictions, '[test] Classifier Predictions')
        plot_wave(predictions_train, '[train] Classifier predicitions')

        print 'classifier prediction size: ', predictions.shape
        print '[test] number of anomalies detected: ', len(predictions[np.where(predictions > 0)])
        print '[train] number of anomalies detected: ', len(predictions_train[np.where(predictions_train > 0)])
        print 'number of anomalies originally: ', source.get_noise_count()

    plt.show()
