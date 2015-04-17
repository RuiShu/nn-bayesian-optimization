"""
@Author: Rui Shu
@Date: 4/11/15

Finds the global maxima of a costly function in a parallelized setting. Runs
optimizer.py in parallel with with several worker nodes that evaluates the costly
function in batch. 

Run as: mpiexec -np 4 python parallel_optimizer.py
where 4 is the number of available processes

Framework: 
Master -- handles the Optimizer object (which takes prior data,
interpolates based on a neural network-linear regression model, and selects the
next set of points to query). Tells worker nodes which points to query.

Worker -- compute the costly function. Returns function evaluation.

Trainer -- in charge of handling the neural network training. 
"""

from mpi_definitions import *
import mpi_master as master
import mpi_worker as worker
import mpi_trainer as trainer

# Check that we have the right number of processes
if size < 3 and not rank == MASTER:
    quit()
elif size < 3:
    print("MASTER: Need at least three processes running.")
    quit()

# Print status of mpi 
print "THE RANK IS: %d, with total size: %d" % (rank, size)

# Setting
lim_domain = np.array([[-1., -1.],
                       [ 1.,  1.]])
init_size = 50

if rank == MASTER:
    master.master_process(lim_domain, init_size)
elif rank == TRAINER:
    trainer.trainer_process()
else:
    worker.worker_process(lim_domain, rank)
