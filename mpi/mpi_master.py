"""
@Author: Rui Shu
@Date: 4/11/15

Master -- handles the Optimizer object (which takes prior data,
interpolates based on a neural network-linear regression model, and selects the
next set of points to query). Tells worker nodes which points to query.
"""

from mpi_definitions import *
import time 

def contains_row(x, X):
    """ Checks if the row x is contained in matrix X
    """
    for i in range(X.shape[0]):
        if all(X[i,:] == x):
            return True

    return False

def master_process(print_statements):
    file_record = open("data/mpi_time_data.csv", "a") # record times for experiment
    from learning_objective.hidden_function import evaluate, true_evaluate, get_settings
    import random
    import utilities.optimizer as op

    print "MASTER: starting with %d workers" % (size - 1)

    # Setup
    t1 = time.time()            # Get amount of time taken
    num_workers = size - 1      # Get number of workers
    closed_workers = 0          # Get number of workers EXIT'ed

    # Get settings relevant to the hidden function being used
    lim_domain, init_size, additional_query_size, init_query, domain, selection_size = get_settings()
    
    # Acquire an initial data set
    dataset = None
    init_assigned = 0           # init query counters
    init_done = 0

    print "Randomly query a set of initial points... ",

    while init_done < init_size:
        # Get a worker (trainer does not initiate conversation with master)
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == WORKER_READY:
            if init_assigned < init_size:
                # Send a (m,) array query to worker
                comm.send(init_query[init_assigned, :], dest=source, tag=SEND_WORKER)
                init_assigned += 1

            else:
                # No more intial work available. Give random work
                comm.send(domain[random.choice(range(domain.shape[0])), :], 
                          dest=source, tag=SEND_WORKER)

        if tag == WORKER_DONE:
            # data is a (1, m) array
            if dataset == None: 
                dataset = data

            else:
                dataset = np.concatenate((dataset, data), axis=0)

            if contains_row(data[0, :-1], init_query):
                init_done += 1

            if print_statements:
                string1 = "MASTER: Init queries done: %3d. " % init_done
                string2 = "Submission from WORKER %2d is: " % source
                print string1 + string2 + str(data)

    print "Complete initial dataset acquired"

    # NN-LR based query system
    optimizer = op.Optimizer(dataset, domain)
    optimizer.train()

    # Select a series of points to query
    selected_points = optimizer.select_multiple(selection_size) # (#points, m) array

    # Set counters
    listen_to_trainer = True
    trainer_is_ready = True     # Determines if trainer will be used
    trainer_index = 0   # Keeps track of data that trainer doesn't have
    selection_index = 0         # Keeps track of unqueried selected_points 
    queries_done = 0            # Keeps track of total queries done
    queries_total = additional_query_size

    t0 = time.time()

    print "Performing optimization..."

    while closed_workers < num_workers:
        if selection_index == selection_size:
            # Update optimizer's dataset and retrain LR
            optimizer.retrain_LR()                            
            selected_points = optimizer.select_multiple(selection_size) # Select new points
            selection_size = selected_points.shape[0]     # Get number of selected points
            selection_index = 0                           # Restart index
            
        if queries_done < queries_total and trainer_is_ready and (dataset.shape[0] - trainer_index - 1) >= 100:
            # Trainer ready and enough new data for trainer to train a new NN.
            if print_statements:
                print "MASTER: Trainer has been summoned"

            comm.send(dataset[trainer_index: -1, :], dest=TRAINER, tag=SEND_TRAINER)
            trainer_index = dataset.shape[0] - 1
            trainer_is_ready = not trainer_is_ready # Trainer is not longer available.

        if queries_done >= queries_total and trainer_is_ready:
            comm.send("MASTER has fired Trainer", dest=TRAINER, tag=EXIT_TRAINER)

        # Check for data from either worker or trainer
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source() 
        tag = status.Get_tag()         

        if tag == WORKER_READY:
            if queries_done < queries_total:
                comm.send(selected_points[selection_index, :], 
                          dest=source, tag=SEND_WORKER) 
                selection_index += 1

            else:
                comm.send(None, dest=source, tag=EXIT_WORKER)

        elif tag == WORKER_DONE:
            dataset = np.concatenate((dataset, data), axis=0) # data is (m, 1) array
            optimizer.update_data(data)                       # add data to optimizer
            queries_done += 1                                 
            
            if print_statements:
                string1 = "MASTER: Queries done: %3d. " % queries_done
                string2 = "Submission from Worker %2d: " % source
                print string1 + string2 + str(data)
            else:
                # Print some sort of progress:
                if queries_done % (queries_total/10) == 0:
                    print "%.3f completion..." % ((1.*queries_done)/queries_total)

            if queries_done <= queries_total:
                info = "%.3f," % (time.time()-t0)
                file_record.write(info)

        elif tag == TRAINER_DONE:
            if listen_to_trainer:
                if print_statements:
                    print "MASTER: Updating neural network"

                W, B, architecture = data
                optimizer.update_params(W, B, architecture)

            trainer_is_ready = not trainer_is_ready 

        elif tag == EXIT_WORKER or tag == EXIT_TRAINER:
            closed_workers += 1 

    file_record.write("NA\n")
    file_record.close()
    t2 = time.time()
    print "MASTER: Total update time is: %3.3f" % (t2-t1)
    print "Best evaluated point is:"
    print dataset[np.argmax(dataset[:, -1]), :]
    print "MASTER: Predicted best point is:"
    optimizer.retrain_LR()
    domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()
    index = np.argmax(pred[:, 0])
    print np.concatenate((np.atleast_2d(domain[index, :]), np.atleast_2d(pred[index, 0])), axis=1)[0, :]
