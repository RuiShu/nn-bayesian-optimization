import numpy as np
import statsmodels.api as sm

class LinearRegressor():

    def __init__(self, dataset):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset

    def predict(self, test_X):
        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]

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

if __name__ == "__main__":
    # Settings
    lim_x        = [-1, 1]                                     # x range for univariate data
    nobs         = 100                                         # number of observed data
    architecture = (1, 50, 50, nobs-2 if nobs < 50 else 50, 1) # Define NN layer architecture
    g            = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2 + np.random.randn()/10 # Define the hidden function
    dataset_X = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    linear_regressor = LinearRegressor(dataset)
    test_X = np.asarray([[i] for i in np.linspace(-lim_x[0], lim_x[1], 100)])
    linear_regressor.predict(test_X)


