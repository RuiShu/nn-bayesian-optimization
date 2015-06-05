# For tahoe server:
# import sys
# sys.path.append('/usr/lib64/python2.7/site-packages/mpich')

from mpi4py import MPI
import numpy as np

MASTER        = 0
TRAINER       = 1
SEND_WORKER   = 3  # Send by Master: gives Worker some work
SEND_TRAINER  = 4  # Send by Master: gives Trainer more data
EXIT_WORKER   = 5  # Send by Master/Worker: tells Worker to quit
EXIT_TRAINER  = 6  # Send by Master/Trainer: tells Trainer to quit
WORKER_READY  = 7  # Send by Worker: tells Master to give work
WORKER_DONE   = 8  # Send by Worker: tells Master the work is done
TRAINER_READY = 9  # Send by Trainer: tells Master to give work
TRAINER_DONE  = 10 # Send by Trainer: gives Master new neural network

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
