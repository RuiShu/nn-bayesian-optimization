import numpy as np
from numpy import atleast_2d as vec
from scipy.stats import multivariate_normal


def gaussian_mix(query):
    # Assign multivariate gaussians to be present in the space.
    gaussians = [multivariate_normal(mean = [0.9, 0.1], cov = [[.05, 0], [0, .05]])]
    gaussians.append(multivariate_normal(mean = [0.9, 0.9], cov = [[0.07, 0.01], [0.01, .07]]))
    gaussians.append(multivariate_normal(mean = [0.15, 0.7], cov = [[.03, 0], [0, .03]]))
    # Initialize initial value.
    value = 0.0
    # Iterate through each gaussian in the space.
    for j in xrange(len(gaussians)):
        value += gaussians[j].pdf(query)

    # Take the average.
    gaussian_function = value / len(gaussians)
    return vec(gaussian_function) # vec(np.array([query.ravel(), gaussian_function]).ravel())


if __name__ == "__main__":
    X = gaussian_mix(np.array([0.5, 0.5]))
    print X
    print X.shape


