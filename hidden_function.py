import numpy as np
import time

def evaluate(query):
    noiseless_g  = lambda x: 10*np.sin(x) - x
    g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function

    
    dataset_X = np.atleast_2d(query)
    print dataset_X
    dataset_Y = np.asarray([[g(dataset_X[0, :])[0]]])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)
    return dataset


    
