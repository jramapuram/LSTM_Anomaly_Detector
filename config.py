__author__ = 'jramapuram'

class config():
    def __init__(self, num_periods=16, num_test_periods=4, max_features=20000
                 , input_dim=128, hidden_dim=64, batch_size=64
                 , activation='tanh', inner_activation='tanh'
                 , initialization='glorot_normal'
                 , model_file='weights_%din_%dhid_%dbatch_%depochs.dat'
                 , optimizer='adam', loss='mean_squared_error', max_epochs=10
                 , truncate_gradient=1, return_sequences=False):
        self.num_periods = num_periods
        self.num_test_periods = num_test_periods
        self.max_features = max_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences
        self.model_file = model_file
        # options: 'glorot_normal', 'glorot_uniform', 'normal', 'he_normal', 'he_uniform'
        self.initialization = initialization
        self.inner_init = inner_activation
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.noise_count = 0

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "num_periods":self.num_periods,
            "num_test_periods":self.num_test_periods,
            "noise_count":self.noise_count,
            "max_features":self.max_features,
            "hidden_dim":self.hidden_dim,
            "initialization":self.initialization,
            "inner_init":self.inner_init,
            "activation":self.activation,
            "optimizer":self.optimizer,
            "loss":self.loss,
            "model_file":self.model_file,
            "batch_size":self.batch_size,
            "max_epochs":self.max_epochs,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}