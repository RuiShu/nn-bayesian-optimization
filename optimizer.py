import numpy as np
import neural_net as nn

class Optimizer(object):

    def __init__(self, dataset):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        nobs = dataset.shape[0]
        self.__architecture = (1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
        self.__feature_extractor = nn.NeuralNet(self.__architecture, dataset)

    def select(self):
        self.__feature_extractor.train()
        train_X = self.__dataset[:, :-1]
        train_features = self.__feature_extractor.extract_features(train_X)
        

if __name__ == "__main__":
    # Settings
    lim_x        = [-1, 1]                                     # x range for univariate data
    nobs         = 100                                         # number of observed data
    architecture = (1, 50, 50, nobs-2 if nobs < 50 else 50, 1) # Define NN layer architecture
    g            = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2 + np.random.randn()/10 # Define the hidden function

    dataset_X = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    optimizer = Optimizer(dataset)
    optimizer.select()

