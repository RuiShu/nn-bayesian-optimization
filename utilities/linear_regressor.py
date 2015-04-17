"""
@Author: Rui Shu
@Date: 4/11/15

Performs linear regression and returns the confidence interval. 
"""

import numpy as np
import statsmodels.api as sm
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt

class LinearRegressor():

    def __init__(self, dataset):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset

    def predict(self, test_X):
        train_X = self.__dataset[:, :-1]
        train_X = sm.add_constant(train_X)
        train_Y = self.__dataset[:, -1:]

        test_X = sm.add_constant(np.atleast_2d(test_X))
        XX_inv,_,_,_ = np.linalg.lstsq(np.dot(train_X.T, train_X), 
                                       np.identity(train_X.shape[1]))
        beta = np.dot(np.dot(XX_inv, train_X.T), train_Y)
        train_pred = np.dot(train_X, beta)
        
        # Confidence interval
        sig = (np.linalg.norm(train_Y-train_pred)**2/(train_X.shape[0]-train_X.shape[1]+1))**0.5

        s = []
        for row in range(test_X.shape[0]):
            x = test_X[[row], :]
            s.append(sig*(1 + np.dot(np.dot(x, XX_inv), x.T))**0.5)
            
        s = np.reshape(np.asarray(s), (test_X.shape[0], 1))

        test_pred = np.dot(test_X, beta)
        hi_ci = test_pred + 2*s
        lo_ci = test_pred - 2*s

        return test_pred, hi_ci, lo_ci

    def predict_reg(self, test_X):
        clf = sklm.Lasso(alpha=1, fit_intercept=False)
        clf.fit(self.__dataset[:, :-1], self.__dataset[:, -1:])
        print np.atleast_2d(clf.coef_)
        pred =clf.predict(test_X)
        pred = np.atleast_2d(pred).T
        return pred, pred, pred
        

if __name__ == "__main__":
    # Settings
    lim_x        = [-10, 10]                                     # x range for univariate data
    nobs         = 50                                         # number of observed data
    g            = lambda x: 2*x + 3 + np.random.randn()*2 # Define the hidden function
    noiseless_g            = lambda x: 2*x + 3 # Define the hidden function

    dataset_X1 = np.asarray([[i] for i in np.linspace(lim_x[0]/10-3, lim_x[1]/10-3, nobs)], dtype=np.float32) # Uniform sampling
    dataset_X2 = np.asarray([[i] for i in np.linspace(lim_x[0]/10+3, lim_x[1]/10+3, nobs)], dtype=np.float32) # Uniform sampling
    dataset_X = np.concatenate((dataset_X1, dataset_X2), axis=0)
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    linear_regressor = LinearRegressor(dataset)
    domain = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], 100)])
    pred, hi_ci, lo_ci = linear_regressor.predict(domain)
    # linear_regressor.predict_reg(test_X)

    ax = plt.gca()
    true_func = np.asarray([[i, noiseless_g(i)] for i in np.linspace(lim_x[0], lim_x[1], 100)], dtype=np.float32)
    plt.plot(true_func[:, 0], true_func[:, 1], 'k', label='true', linewidth=4) # true plot
    plt.plot(domain, pred, 'c--', label='LR regression', linewidth=7)
    plt.plot(domain, hi_ci, 'g--', label='ci')
    plt.plot(domain, lo_ci, 'g--')
    plt.plot(dataset[:,:-1], dataset[:, -1:], 'rv', label='training', markersize=7.)
    plt.xlabel('Input space')
    plt.ylabel('Output space')
    plt.title("LR regression")
    plt.legend()
    plt.show()
