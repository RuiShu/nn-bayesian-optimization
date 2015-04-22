import time
import numpy as np
import neural_net as nn
import linear_regressor as lm
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from learning_objective.hidden_function import evaluate, true_evaluate
import statsmodels.api as sm

class Optimizer(object):

    def __init__(self, dataset, domain):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        nobs = dataset.shape[0]
        self.__architecture = (domain.shape[1], 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
        self.__domain = domain

    def train(self):
        """ Using the stored dataset and architecture, trains the neural net to 
        perform feature extraction, and the linear regressor to perform prediction
        and confidence interval computation.
        """
        neural_net = nn.NeuralNet(self.__architecture, self.__dataset)
        neural_net.train()
        self.__W, self.__B = neural_net.extract_params()
        self.__nn_pred = neural_net.e.network(self.__domain)

        # Extract features
        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]
        train_features = self.extract_features(train_X)
        domain_features = self.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False)
        linear_regressor.train()
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

    def retrain_NN(self):
        neural_net = nn.NeuralNet(self.__architecture, self.__dataset)
        neural_net.train()
        self.__W, self.__B = neural_net.extract_params()

    def retrain_LR(self):
        """ After the selected point (see select()) is queried, insert the new info
        into dataset. Depending on the size of the dataset, the module decides whether
        to re-train the neural net (for feature extraction). 
        A new interpolation is then constructed.

        Keyword arguments:
        new_data -- a 1 by (m+1) array that forms the matrix [X, Y]
        """

        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]

        # Extract features
        train_features = self.extract_features(train_X)
        domain_features = self.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset)
        linear_regressor.train()
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

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
        # -(min(train_Y) - prediction)/sig # finding max 
        # ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
        gamma = (prediction - np.max(train_Y)) / sig 
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
        gamma = (prediction - np.max(train_Y)) / sig
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

        return np.atleast_2d(self.__domain[select_indices, :])

    def check_point(self, selected_index, order):
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = (hi_ci[selected_index] - prediction[selected_index])/2
        z_score = abs(prediction[order] - prediction[selected_index])/sig

        return (stats.norm.cdf(-z_score)*2) < 0.5

    def update_data(self, new_data):
        nobs = self.__dataset.shape[0]
        if nobs < 50:
            nobs += new_data.shape[0]
            self.__architecture = (self.__domain.shape[1], 50, 50, nobs - 2 if nobs < 50 else 50, 1)
        
        self.__dataset = np.concatenate((self.__dataset, new_data), axis=0)
    
    def update_params(self, W, B, architecture):
        self.__W = W
        self.__B = B
        self.__architecture = architecture

    def get_prediction(self):
        return (self.__domain, self.__pred, self.__hi_ci, 
                self.__lo_ci, self.__nn_pred, self.__ei, self.__gamma)

    def get_dataset(self):
        return self.__dataset

if __name__ == "__main__":
    t1 = time.time()
    random.seed(42)
    # Settings
    lim_domain = np.array([[-1, -1],
                           [ 1,  1]]) 
    nobs         = 50                                         # number of observed data

    # Create dataset
    dataset_X = np.random.uniform(-1, 1, size=(nobs, lim_domain.shape[1]))
    dataset = evaluate(dataset_X[0, :], lim_domain)

    for i in range(1, dataset_X.shape[0]):
        dataset = np.concatenate((dataset, evaluate(dataset_X[i, :], lim_domain)))

    domain = dataset[:, :-1]

    # Instantiate Optimizer
    optimizer = Optimizer(dataset, domain)
    optimizer.train()
    selected_points = optimizer.select_multiple()
    selection_index = 0
    selection_size = selected_points.shape[0]

    # Select a point
    for _ in range(50):
        if selection_index == selection_size:
            optimizer.retrain_LR()
            selected_points = optimizer.select_multiple()
            selection_size = selected_points.shape[0]
            selection_index = 0
            
        new_data = evaluate(selected_points[selection_index, :], lim_domain)
        print "New evaluation: " + str(new_data)
        selection_index += 1
        optimizer.update_data(new_data)


    dataset = optimizer.get_dataset()
    selected_point = optimizer.select_multiple()[0, :]

    domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()

    t2 = time.time()

    print "optimizer: Total update time is: %3.3f" % (t2-t1)

    # Plot results
    if False:
        ax = plt.gca()
        plt.plot(domain, pred, 'c', label='NN-LR regression', linewidth=3)
        plt.plot(domain, nn_pred, 'r', label='NN regression', linewidth=3)
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
        figpath = 'figures/seq_regression_' + str(int(time.time())) + '.eps'
        plt.savefig(figpath, format='eps', dpi=2000)
        # plt.show()
