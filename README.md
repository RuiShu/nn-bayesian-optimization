# Adaptive Neural Network Representations for Parallel and Scalable Bayesian Optimization

**Neural Network Bayesian Optimization** is function optimization technique inpsired by the work of:
> Jasper Snoek, et al <br>
> Scalable Bayesian Optimization Using Deep Neural Networks <br>
> http://arxiv.org/abs/1502.05700

This repository contains the python code written by James Brofos and Rui Shu of a modified approach that continually retrains the neural network underlying the optimization technique, and implements the technique within a parallelized setting for improved speed performance. 

Motivation
----------
The success of most machine learning algorithms is dependent the proper tuning of the hyperparameters. A popular technique for hyperparameter tuning is Bayesian optimization, which canonically uses a Gaussian process to interpolate the hyperparameter space. The computation time for GP-based Bayesian optimization, however, grows rapidly with respect to sample size (the number of tested hyperparameters) and quickly becomes very time consuming, if not all together intractable. Fortunately, a neural network is capable of mimicking the behavior of a Guassian process whilst providing a significant reduction in computation time. 

Dependencies
------------
This code requires:

* Python 2.7
* MPI (and [MPI4Py](http://mpi4py.scipy.org/))
* [Numpy](http://www.numpy.org/) 
* [Scipy](http://www.scipy.org/)
* [Theano](http://deeplearning.net/software/theano/)
* [Theanets](http://theanets.readthedocs.org/en/stable/)
* [Statsmodels](http://statsmodels.sourceforge.net/devel/)
* [Matplotlib](http://matplotlib.org/)
* [pyGPs](http://www-ai.cs.uni-dortmund.de/weblab/static/api_docs/pyGPs/)

Code Execution
--------------
To run the code from the home directory in parallel with 4 cores, simply call mpiexec:
```       
mpiexec -np 4 python -m mpi.mpi_optimizer
```

To run a sequential version of the code:
```       
python -m sequential.seq_optimizer
```

To run the gaussian process version of Bayesian optimization:
```
python -m sequential.seq_gaussian_process
```

**Sample output**:
```
Randomly query a set of initial points...  Complete initial dataset acquired
Performing optimization... 
0.100 completion...
0.200 completion...
0.300 completion...
0.400 completion...
0.500 completion...
0.600 completion...
0.700 completion...
0.800 completion...
0.900 completion...
1.000 completion...
Sequential gp optimization task complete.
Best evaluated point is:
[-0.31226245  3.80792522]
Predicted best point is:
[-0.31226245  3.7755048 ]
```

**Note:** The code, as written, focuses the use of the algorithm on any black-box function. A few common functions are available in `learning_objective`. The chosen function is set in `hidden_function.py`. To really appreciate the time-savings gained by the parallelized code, it is important to realize that evaluating a real-world black-box function (i.e. computing the test performance for an ML algorithm with a given set of hyperparameters) takes time.

This can be simulated by uncommenting the line: `# time.sleep(2)` in `hidden_function.py`.

