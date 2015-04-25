import numpy as np
from numpy import atleast_2d as vec

def gaussian_process(xs):
    return m(xs)

def m(x):
    return vec(-3*x*(1.5+3*x)*(3*x-1.5)*(3*x-2))

if __name__ == "__main__":
    print gaussian_process(0)
