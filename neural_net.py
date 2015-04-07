import numpy as np
import theanets
import statsmodels.api as sm

class NeuralNet(object):
    
    def __init__(self, architecture, dataset):
        """Initialization of NeuralNet object
        
        Keyword arguments:
        architecture -- a tuple containing the number of nodes in each layer
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__architecture = architecture
        self.__dataset = dataset
        self.__e = None
    
    def train(self):
        architecture = self.__architecture
        dataset = self.__dataset

        cut = int(0.9 * len(dataset))  # select 90% of data for training, 10% for validation
        idx = range(len(dataset))
        np.random.shuffle(idx)
        
        train = idx[:cut]
        train_set = [dataset[train, :-1], dataset[train, -1:]]
        valid = idx[cut:]
        valid_set = [dataset[valid, :-1], dataset[valid, -1:]]
        
        e = theanets.Experiment(theanets.feedforward.Regressor,
                                layers=architecture,
                                optimize='sgd',
                                activation='tanh', 
                                output_activation='linear',
                                learning_rate=0.01)
        
        e.run(train_set, valid_set)
        self.__e = e
        self.extract_params()

    def extract_params(self):
        architecture = self.__architecture
        e = self.__e
        # Extract parameters
        W = {}
        B = {}
        for i in range(len(architecture)-2):
            W[i] = e.network.params[2*i].get_value()
            B[i] = np.reshape(e.network.params[2*i+1].get_value(), (1, architecture[i+1]))
            
        self.__W = W
        self.__B = B

    def extract_features(self, test_X):
        W = self.__W
        B = self.__B
        architecture = self.__architecture

        # Feedforward into custom neural net
        X = []
        
        for i in range(test_X.shape[0]):
            test_val = test_X[[i], :]
            L = np.tanh(np.dot(test_val, W[0]) + B[0])
            
            for i in range(1, len(architecture)-2):
                L = np.tanh(np.dot(L, W[i]) + B[i])
                
            X.extend(L.tolist())
                
        X = np.asarray(X)
        X = sm.add_constant(X)

        return X

if __name__ == "__main__":
    # Settings
    lim_x        = [-1, 1]                                     # x range for univariate data
    nobs         = 100                                         # number of observed data
    architecture = (1, 50, 50, nobs-2 if nobs < 50 else 50, 1) # Define NN layer architecture
    g            = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2 + np.random.randn()/10 # Define the hidden function

    dataset_X = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    feature_extractor = NeuralNet(architecture, dataset)
    feature_extractor.train()
    train_X = dataset[:, :-1]
    train_features = feature_extractor.extract_features(train_X)
