"""
@Author: Rui Shu
@Date: 4/11/15

Provides a proxy hidden function for running of optimizer and mpi_optimizer
"""

import numpy as np
import time
from gaussian_mix import gaussian_mix as gm

noiseless_g  = lambda x: 10*np.sin(x) - x
g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function

def evaluate(query, lim_domain):
    """ Queries a single point with noise.

    Keyword arguments:
    query      -- a (m,) array. Single point query in input space scaled to unit cube.
    lim_domain -- a (2, m) array. Defines the search space boundaries of the 
                  true input space
    """
    var     = (lim_domain[1, :] - lim_domain[0, :])/2.
    mean    = (lim_domain[1, :] + lim_domain[0, :])/2.
    query   = np.atleast_2d(query)      # Convert to (1, m) array
    X       = query*var + mean          # Scale query to true input space
    dataset = np.concatenate((query, gm(X)), axis=1)
    
    # time.sleep(0.5)
    return dataset
    
def evaluate_alt(query, lim_domain):
    """ Queries a single point with noise.

    Keyword arguments:
    query      -- a (m,) array. Single point query in input space scaled to unit cube.
    lim_domain -- a (2, m) array. Defines the search space boundaries of the 
                  true input space
    """
    var     = (lim_domain[1, :] - lim_domain[0, :])/2.
    mean    = (lim_domain[1, :] + lim_domain[0, :])/2.
    query   = np.atleast_2d(query)      # Convert to (1, m) array
    X       = query*var + mean          # Scale query to true input space
    Y       = np.atleast_2d(g(X[0, 0])) # Compute output
    dataset = np.concatenate((query, Y), axis=1)  

    # time.sleep(0.5)
    return dataset
    
def true_evaluate(query, lim_domain):
    """ Queries a single point without noise.

    Keyword arguments:
    query      -- a (m,) array. Single point query in input space scaled to unit cube.
    lim_domain -- a (2, m) array. Defines the search space boundaries of the 
                  true input space
    """
    var     = (lim_domain[1, :] - lim_domain[0, :])/2.
    mean    = (lim_domain[1, :] + lim_domain[0, :])/2.
    query   = np.atleast_2d(query)                # Convert to (1, m) array
    X       = query*var + mean                    # Scale query to true input space
    Y       = np.atleast_2d(noiseless_g(X[0, 0])) # Compute output
    dataset = np.concatenate((query, Y), axis=1)  

    # time.sleep(0.5)
    return dataset
