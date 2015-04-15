import numpy as np
import time

def evaluate(query):
    noiseless_g  = lambda x: 10*np.sin(x) - x
    g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function

    
    dataset_X = np.atleast_2d(query)
    dataset_Y = np.asarray([[g(dataset_X[0, :])[0]]])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    # time.sleep(0.5)
    
    return dataset
    
def true_evaluate(query):
    g            = lambda x: 10*np.sin(x) - x
    
    dataset_X = np.atleast_2d(query)
    dataset_Y = np.asarray([[g(dataset_X[0, :])[0]]])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    # time.sleep(0.5)
    
    return dataset
    
