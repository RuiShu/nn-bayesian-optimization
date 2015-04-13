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
plot_it = True
def contains_row(x, X):
    """ Checks if the row x is contained in matrix X
    """

    for i in range(X.shape[0]):
        if X[i,:] == x:
            return True

    return False

def master_process(lim_x, init_size):
    import mpi_master
    import random
    import matplotlib.pyplot as plt
    import optimizer as op

    num_workers = size - 1
    closed_workers = 0
    trainer_is_ready = True

    print "Master starting with %d workers" % num_workers

    init_query = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], init_size)], dtype=np.float32) # Uniform sampling
    domain = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], 100)])

    # Acquire an initial data set
    tasks_assigned = 0
    tasks_done = 0
    tasks_total = init_size
    dataset = None
    
    # Initial query
    while tasks_done < init_size:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) # receive worker info
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == WORKER_READY:
            if tasks_assigned < tasks_total:
                comm.send(init_query[tasks_assigned, :], dest=source, tag=SEND_WORKER)
                tasks_assigned += 1
            else:
                print "MASTER: No more intial work available. Give random work."
                comm.send(domain[random.choice(range(domain.shape[0])), :], dest=source, tag=SEND_WORKER)

        if tag == WORKER_DONE:
            if dataset == None:
                dataset = data
            else:
                dataset = np.concatenate((dataset, data), axis=0)

            if contains_row(data[0, :-1], init_query):
                tasks_done += 1

    print "Complete initial dataset acquired"
    print dataset

    # Principled query
    optimizer = op.Optimizer(dataset, domain)
    optimizer.train()
    selected_points = optimizer.select_multiple()
    selection_size = selected_points.shape[0]
    selection_index = 0
    trainer_dataset_index = 0
    tasks_done = 0
    tasks_total = 0

    while False:
    # while closed_workers < num_workers:
        if selection_index == selection_size:
            selected_points = optimizer.select_multiple()
            selection_index = 0
            
        if trainer_is_ready:
            # If trainer is ready, keep shoving data at him, if there is data to be shoved
            additional_dataset = dataset[trainer_dataset_index: -1, :]
            comm.send(additional_dataset, dest=TRAINER, tag=SEND_TRAINER)
            trainser_dataset_index = dataset.shape[0] - 1
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
                comm.send(selected_points[selection_index], dest=source, tag=SEND_WORKER)
                selection_index += 1
            else:
                print "Master: Killing Worker"
                comm.send(None, dest=source, tag=EXIT_WORKER)

        elif tag == WORKER_DONE:
            # If worker is done, tally the total amount of work done. 
            dataset = np.concatenate((dataset, data), axis=0)
            optimizer.update_data(data)
            tasks_done += 1
            print "Number of total tasks: %d" % tasks_done

        elif tag == TRAINER_DONE:
            # If trainer is done, store what trainer did. 
            optimizer.update_feature_extractor(data)
            train_is_ready = not trainer_is_ready

        elif tag == EXIT_WORKER or tag == EXIT_TRAINER:
            # If worker has exited, tally up number of closed workers.
            closed_workers += 1

    print "Master is done."

    # Plot results
    if plot_it:
        optimizer.train()
        selected_point = optimizer.select_multiple()[0, :]
        domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()
        ax = plt.gca()
        plt.plot(domain, pred, 'c--', label='NN-LR regression', linewidth=7)
        plt.plot(domain, nn_pred, 'r--', label='NN regression', linewidth=7)
        plt.plot(domain, hi_ci, 'g--', label='ci')
        plt.plot(domain, lo_ci, 'g--')
        # plt.plot(domain, ei, 'b--', label='ei')
        # plt.plot(domain, gamma, 'r', label='gamma')
        plt.plot([selected_point, selected_point], [ax.axis()[2], ax.axis()[3]], 'r--',
                 label='EI selection')
        plt.plot(dataset[:,:-1], dataset[:, -1:], 'rv', label='training', markersize=7.)
        plt.xlabel('Input space')
        plt.ylabel('Output space')
        plt.title("NN-LR regression")
        plt.legend()
        plt.show()


def trainer_process():
    import neural_net as nn
    nobs = 0
    dataset = None
    untrained_data_count = 0

    while True:
        new_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if untrained_data_count > 100:
            architecture = (1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
            feature_extractor = nn.NeuralNet(architecture, dataset)
            comm.send(feature_extractor, dest=0, tag=TRAINER_DONE)

        if tag == SEND_TRAINER:
            if dataset == None:
                dataset = new_data
            else:
                dataset = np.concatenate(dataset, new_data)

            untrained_data_count += new_data.shape[0]
            nobs = dataset.shape[0]
            
        elif tag == EXIT_TRAINER:
            print "TRAINER: Commiting suicide"
            break

    comm.send(None, dest=0, tag=EXIT_TRAINER) # Suicide complete


def worker_process(rank):
    from hidden_function import evaluate

    while True:
        comm.send("WORKER is ready", dest=0, tag=WORKER_READY)    # tell Master node that I need a new query
        query  = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_WORKER:
            statement = "WORKER %3d: The query is: " % rank
            print statement + str(query)  
            result = evaluate(query)
            comm.send(result, dest=0, tag=WORKER_DONE)

        elif tag == EXIT_WORKER:
            # Worker dies!
            print "WORKER: Commiting suicide"
            break

    comm.send(None, dest=0, tag=EXIT_WORKER) # Suicide complete
    
if rank == MASTER:                         # MASTER NODE
    # Settings
    lim_x        = [-6, 4]                                     # x range for univariate data
    init_size = 50
    master_process(lim_x, init_size)

elif rank == TRAINER:
    trainer_process()
else:
    worker_process(rank)
