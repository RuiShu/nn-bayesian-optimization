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

print "THE RANK IS: %d, with total size: %d" % (rank, size)

def master_process():
    import mpi_master
    master = mpi_master.Master(size)

    tasks_done = 0
    tasks_total = 4
    num_workers = size - 1
    closed_workers = 0
    print "Master starting with %d workers" % num_workers
    trainer_is_ready = True

    while closed_workers < num_workers:
        if trainer_is_ready:
            # If trainer is ready, keep shoving data at him, if there is data to be shoved
            comm.send("Master has sent Trainer something", dest=TRAINER, tag=SEND_TRAINER)
            trainer_is_ready = not trainer_is_ready

        if not (tasks_done < tasks_total):
            print "Master: Killing Trainer"
            comm.send("Master has fired Trainer", dest=TRAINER, tag=EXIT_TRAINER)

        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) # receive worker info
        source = status.Get_source() 
        tag = status.Get_tag()          # check what worker's request is

        if tag == WORKER_READY:
            # If worker is ready, check if you have work left. If all work is completed,
            # tell worker to exit. If work is available, give it to worker.
            if tasks_done < tasks_total:
                comm.send("Master has sent Worker something", dest=source, tag=SEND_WORKER)

            else:
                print "Master: Killing Worker"
                comm.send(None, dest=source, tag=EXIT_WORKER)

        elif tag == WORKER_DONE:
            # If worker is done, tally the total amount of work done. 
            tasks_done += 1
            print "Number of total tasks: %d" % tasks_done

        elif tag == TRAINER_DONE:
            # If trainer is done, store what trainer did. 
            train_is_ready = not trainer_is_ready

        elif tag == EXIT_WORKER or tag == EXIT_TRAINER:
            # If worker has exited, tally up number of closed workers.
            closed_workers += 1

    print "Master is done."

    
def trainer_process():
    while True:
        new_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_TRAINER:
            print new_data
            # Does its thing
            comm.send("Trainer is done", dest=0, tag=TRAINER_DONE)

        elif tag == EXIT_TRAINER:
            print "Trainer: Commiting suicide"
            break

    comm.send(None, dest=0, tag=EXIT_TRAINER) # Suicide complete

def worker_process():
    while True:
        comm.send("Worker is ready", dest=0, tag=WORKER_READY)    # tell Master node that I need a new query
        query  = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_WORKER:
            # The worker does its thing
            print query
            comm.send("Worker is done", dest=0, tag=WORKER_DONE)

        elif tag == EXIT_WORKER:
            # Worker dies
            print "Worker: Commiting suicide"
            break

    comm.send(None, dest=0, tag=EXIT_WORKER) # Suicide complete

    
if rank == MASTER:                         # MASTER NODE
    master_process()
elif rank == TRAINER:
    trainer_process()
else:
    worker_process()
