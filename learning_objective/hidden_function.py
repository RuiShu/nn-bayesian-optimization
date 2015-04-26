"""
@Author: Rui Shu
@Date: 4/11/15

Provides a proxy hidden function for running of optimizer and mpi_optimizer
"""

# To change between hidden funcs:
# Make sure lim_domain is set correctly
# Make sure r linspace is set correctly
# Make sure r meshgrid is set correctly
# Make sure set gm v. gp v. hm

import numpy as np
import time
from gaussian_mix import gaussian_mix as gm
from hartmann import hartmann as hm
from gaussian_process import gaussian_process as gp

HM = 0
GP = 1
GM = 2

# Set it
method = HM

def get_settings(lim_domain_only=False):
    # Settings
    if method == HM:
        lim_domain = np.array([[0., 0., 0., 0.],
                               [ 1.,  1., 1., 1.]])
    elif method == GM:
        lim_domain = np.array([[-1., -1.],
                               [ 1.,  1.]])
    elif method == GP:
        lim_domain = np.array([[-1.],
                               [ 1.]])

    if lim_domain_only:
        return lim_domain

    init_size = 50
    additional_query_size = 400
    selection_size = 1

    # Get initial set of locations to query
    init_query = np.random.uniform(-1, 1, size=(init_size, lim_domain.shape[1]))

    # WARNING. SET THE THING YOURSELF FOR NOW.
    if method == HM:
        r = np.linspace(-1, 1, 15)
        X = np.meshgrid(r, r, r, r)
    elif method == GM:
        r = np.linspace(-1, 1, 50)
        X = np.meshgrid(r, r)

    if method == GP:
        domain = np.atleast_2d(np.linspace(-1, 1, 2500)).T
    else:
        xx = np.atleast_2d([x.ravel() for x in X]).T
        domain = np.atleast_2d(xx[0])
        for i in range(1, xx.shape[0]):
            domain = np.concatenate((domain, np.atleast_2d(xx[i])), axis=0)

    return lim_domain, init_size, additional_query_size, init_query, domain, selection_size

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

    if method == GM:
        dataset = np.concatenate((query, gm(X) + np.random.randn()/100), axis=1)
    elif method == HM:
        dataset = np.concatenate((query, hm(X) + np.random.randn()/100), axis=1)
    elif method == GP:
        dataset = np.concatenate((query, gp(X) + np.random.randn()/100), axis=1)
    
    # time.sleep(2)
    return dataset

def true_evaluate(query, lim_domain):
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

    if method == GM:
        dataset = np.concatenate((query, gm(X)), axis=1)
    elif method == HM:
        dataset = np.concatenate((query, hm(X)), axis=1)
    elif method == GP:
        dataset = np.concatenate((query, gp(X)), axis=1)
    
    # time.sleep(2)
    return dataset


""" Alternative functions
"""
noiseless_g  = lambda x: 10*np.sin(x) - x
g            = lambda x: noiseless_g(x) + np.random.randn()/10 # Define the hidden function


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
    
def true_evaluate_alt(query, lim_domain):
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
