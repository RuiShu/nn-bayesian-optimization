import numpy as np
import time

def evaluate(query, lim_domain):
    """ Query is always a single query
    """
    noiseless_g  = lambda x: 10*np.sin(x) - x
    g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function

    # Transform to cube: scaled = (val - mean)/var. 
    # Transform back: *scaled*var) + mean
    var = (lim_domain[1, :] - lim_domain[0, :])/2.
    mean = (lim_domain[1, :] + lim_domain[0, :])/2.
    query = np.atleast_2d(query)
    dataset_X = query*var + mean
    dataset_Y = np.atleast_2d(g(dataset_X)[0, 0])
    dataset = np.concatenate((query, dataset_Y), axis=1)
    # time.sleep(0.5)
    return dataset
    
def true_evaluate(query, scale):
    g            = lambda x: 10*np.sin(x) - x
    
    dataset_X = np.atleast_2d(query)
    dataset_Y = np.asarray([[g(scale*dataset_X[0, :])[0]]])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    # time.sleep(0.5)
    
    return dataset
    
