import numpy as np

def contains_row(x, X):
    """ Checks if the row x is contained in matrix X
    """

    for i in range(X.shape[0]):
        if X[i,:] == x:
            return True

    return False

if __name__ == "__main__":
    domain = np.asarray([[i] for i in np.linspace(0, 10, 5)])
    x = np.asarray([[0., 0.], [1, 1]])
    print domain
    print x
    print contains_row(x[:, :-1], domain)
