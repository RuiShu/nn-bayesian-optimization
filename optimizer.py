import numpy as np
import neural_net as nn
import linear_regressor as lm
import scipy.stats as stats
import matplotlib.pyplot as plt

class Optimizer(object):

    def __init__(self, dataset, domain):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        nobs = dataset.shape[0]
        self.__architecture = (1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
        self.__feature_extractor = nn.NeuralNet(self.__architecture, dataset)
        self.__domain = domain

    def train(self):
        self.__feature_extractor.train()
        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]

        # Extract features
        train_features = self.__feature_extractor.extract_features(train_X)
        domain_features = self.__feature_extractor.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset)
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

    def select(self):
        train_Y    = self.__dataset[:, -1:]
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = (hi_ci - prediction)/2
        # gamma = (min(train_Y) - prediction)/sig # finding min
        gamma = -(min(train_Y) - prediction)/sig # finding max
        ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
        index = np.argmax(ei)
        return self.__domain[index, :]

    def get_prediction(self):
        return self.__domain, self.__pred, self.__hi_ci, self.__lo_ci


if __name__ == "__main__":
    # Settings
    lim_x        = [-1, 1]                                     # x range for univariate data
    nobs         = 100                                         # number of observed data
    architecture = (1, 50, 50, nobs-2 if nobs < 50 else 50, 1) # Define NN layer architecture
    g            = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2 + np.random.randn()/10 # Define the hidden function
    noiseless_g  = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2             

    # Create dataset
    dataset_X = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    domain = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], 100)])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)

    # Instantiate Optimizer
    optimizer = Optimizer(dataset, domain)
    optimizer.train()
    selected_point = optimizer.select()
    domain, pred, hi_ci, lo_ci = optimizer.get_prediction()

    # Plot results
    ax = plt.gca()
    true_func = np.asarray([[i, noiseless_g(i)] for i in np.linspace(lim_x[0], lim_x[1], 100)], dtype=np.float32)
    plt.plot(true_func[:, 0], true_func[:, 1], 'k', label='true', linewidth=4) # true plot
    plt.plot(domain, pred, 'c--', label='NN-LR regression', linewidth=7)
    plt.plot(domain, hi_ci, 'g--', label='ci')
    plt.plot(domain, lo_ci, 'g--')
    plt.plot([selected_point, selected_point], [ax.axis()[2], ax.axis()[3]], 'r--',
             label='EI selection')
    plt.plot(dataset_X, dataset_Y, 'rv', label='training', markersize=7.)
    plt.xlabel('Input space')
    plt.ylabel('Output space')
    plt.title("NN-LR regression")
    plt.legend()
    plt.show()
