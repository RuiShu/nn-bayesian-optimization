import time
import numpy as np
import neural_net as nn
import linear_regressor as lm
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from learning_objective.hidden_function import evaluate

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

    def update_feature_extractor(self, feature_extractor):
        self.__feature_extractor = feature_extractor

    def train(self):
        """ Using the stored dataset and architecture, trains the neural net to 
        perform feature extraction, and the linear regressor to perform prediction
        and confidence interval computation.
        """

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
        self.__nn_pred = self.__feature_extractor.e.network(self.__domain)
        # self.__hi_ci = 0*(self.__hi_ci - self.__pred) + self.__pred
        # self.__lo_ci = 0*(self.__lo_ci - self.__pred) + self.__pred

    def select(self):
        """ Selects and returns the point in the domain X that has the max expected
        improvements.
        """

        train_Y    = self.__dataset[:, -1:]
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = (hi_ci - prediction)/2
        # gamma = (min(train_Y) - prediction)/sig # finding min
        # gamma = -(min(train_Y) - prediction)/sig # finding max
        gamma = (prediction - np.max(train_Y)) / sig # -(min(train_Y) - prediction)/sig # finding max        ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
        self.__ei = ei
        index = np.argmax(ei)
        return self.__domain[index, :]

    def select_multiple(self):
        """ Identify multiple points. 
        """
        
        # Rank order by expected improvement
        train_Y    = self.__dataset[:, -1:]
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = abs((hi_ci - prediction)/2)
        # gamma = (min(train_Y) - prediction)/sig # finding min
        gamma = (prediction - np.max(train_Y)) / sig # -(min(train_Y) - prediction)/sig # finding max
        ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))

        if np.max(ei) <= 0:
            sig_order = np.argsort(-sig, axis=0)
            select_indices = sig_order[:5, 0].tolist()
            print "optimizer.py: Pure exploration"
        else:
            ei_order = np.argsort(-1*ei, axis=0)
            select_indices = [ei_order[0, 0]]

            for candidate in ei_order[:, 0]:
                keep = True
                for selected_index in select_indices:
                    keep = keep*self.check_point(selected_index, candidate)
                if keep and ei[candidate, 0] > 0:
                    select_indices.append(candidate)
                if len(select_indices) == 5: # Number of points to select
                    break 

            if len(select_indices) < 5:
                print "optimizer.py: Exploration appended"
                sig_order = np.argsort(-sig, axis=0)
                add_indices = sig_order[:(5-len(select_indices)), 0].tolist()
                select_indices.extend(add_indices)
            else:
                print "optimizer.py: All expected"

        index = np.argmax(ei)
        self.__gamma = gamma
        self.__ei = ei

        # print np.atleast_2d(self.__domain[select_indices, :])
        return np.atleast_2d(self.__domain[select_indices, :])

    def check_point(self, selected_index, order):
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = (hi_ci[selected_index] - prediction[selected_index])/2
        z_score = abs(prediction[order] - prediction[selected_index])/sig

        return (stats.norm.cdf(-z_score)*2) < 0.5

    def update_data(self, new_data):
        self.__dataset = np.concatenate((self.__dataset, new_data), axis=0)
        self.__feature_extractor.update_data(new_data)

    def update(self, new_data=None):
        """ After the selected point (see select()) is queried, insert the new info
        into dataset. Depending on the size of the dataset, the module decides whether
        to re-train the neural net (for feature extraction). 
        A new interpolation is then constructed.

        Keyword arguments:
        new_data -- a 1 by (m+1) array that forms the matrix [X, Y]
        """

        if not (new_data == None):
            self.__dataset = np.concatenate((self.__dataset, new_data), axis=0)

        nobs = self.__dataset.shape[0]

        if nobs < 50:
            # Retrain NN if number of samples is less than 100 
            self.__architecture = (1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
            self.__feature_extractor.update(self.__architecture, new_data)

        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]

        # Extract features
        train_features = self.__feature_extractor.extract_features(train_X)
        domain_features = self.__feature_extractor.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset)
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

    def get_prediction(self):
        return self.__domain, self.__pred, self.__hi_ci, self.__lo_ci, self.__nn_pred, self.__ei, self.__gamma

    def get_dataset(self):
        return self.__dataset

if __name__ == "__main__":
    t1 = time.time()
    random.seed(42)
    # Settings
    lim_x        = [-6, 4]                                     # x range for univariate data
    nobs         = 50                                         # number of observed data
    architecture = (1, 50, nobs-2 if nobs < 50 else 50, 1) # Define NN layer architecture
    # g            = lambda x: np.exp(-x)*np.sin(10*x)*x-10*x**2 + np.random.randn()/10 # Define the hidden function
    # noiseless_g  = lambda x: np.exp(-x)*np.sin(10*x)*x-10*x**2             
    noiseless_g  = lambda x: 10*np.sin(x) - x
    g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function

    # Create dataset
    
    # dataset_X = np.asarray([[i] for i in np.linspace(0, lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    scale = np.max(np.abs(lim_x))

    dataset_X = np.asarray([[np.random.uniform(0, 1)] for _ in range(nobs)],
                            dtype=np.float32) # Random uniform sampling
    dataset = evaluate(dataset_X[0, :], scale)

    for i in range(1, dataset_X.shape[0]):
        dataset = np.concatenate((dataset, evaluate(dataset_X[i, :], scale)))

    domain = np.asarray([[i] for i in np.linspace(-1, 1, 1000)])
    
    # Instantiate Optimizer
    optimizer = Optimizer(dataset, domain)
    optimizer.train()
    selected_points = optimizer.select_multiple()
    selection_index = 0
    selection_size = selected_points.shape[0]

    # Select a point
    for _ in range(50):
        # print "start next point selection: " + str(selected_point)
        # Update
        # new_data = np.atleast_2d(np.concatenate((selected_point, g(selected_point))))
        if selection_index == selection_size:
            optimizer.update()
            selected_points = optimizer.select_multiple()
            selection_size = selected_points.shape[0]
            selection_index = 0
            
        new_data = evaluate(selected_points[selection_index, :], scale)
        print "New evaluation: " + str(new_data)
        selection_index += 1
        optimizer.update_data(new_data)


    # print "optimizer.py: final training"
    # optimizer.train()
    dataset = optimizer.get_dataset()
    selected_point = optimizer.select_multiple()[0, :]

    domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()

    t2 = time.time()

    print "optimizer: Total update time is: %3.3f" % (t2-t1)

    # Plot results
    ax = plt.gca()
    # true_func = np.asarray([[i, noiseless_g(i)] for i in np.linspace(lim_x[0], lim_x[1], 100)], dtype=np.float32)
    # plt.plot(true_func[:, 0], true_func[:, 1], 'k', label='true', linewidth=4) # true plot
    plt.plot(domain, pred, 'c--', label='NN-LR regression', linewidth=7)
    plt.plot(domain, nn_pred, 'r--', label='NN regression', linewidth=7)
    plt.plot(domain, hi_ci, 'g--', label='ci')
    plt.plot(domain, lo_ci, 'g--')
    # plt.plot(domain, ei, 'b--', label='ei')
    # plt.plot(domain, gamma, 'r', label='gamma')
    plt.plot([selected_point, selected_point], [ax.axis()[2], ax.axis()[3]], 'r--',
             label='EI selection')
    plt.plot(dataset[:,:-1], dataset[:, -1:], 'rv', label='training', markersize=7.)
    plt.xlabel('Input space')
    plt.ylabel('Output space')
    plt.title("NN-LR regression")
    plt.legend()
    plt.show()
