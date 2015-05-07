# LSTM Based Anomaly Detection
Hypothetically it should be possible to autoencoder an LSTM in order to build an anomaly detector. 
This proves that this is possible. 

## Example
It is shown that a simple sin wave with added amplitudal noise can be detected if trained with a pure unsupervised LSTM autoencoder. 

The entire signal as it is presented to the network [sliding window] :

![Alt text](/../screenshots/images/sliding_unrolled.png?raw=true "Full signal's sliding window concatinated together")

The signal approximation (mean) :

![Alt text](/../screenshots/images/rolled_up_wave.png?raw=true "Full signal's sliding window mean")

The MSE output of the LSTM detector:

![Alt text](/../screenshots/images/LSTM_classification.png?raw=true "LSTM detector output")

