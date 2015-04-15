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
import time 

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
    from learning_objective.hidden_function import evaluate, true_evaluate
    import mpi_master
    import random
    import matplotlib.pyplot as plt
    import utilities.optimizer as op
    t1 = time.time()
    scale = np.max(np.abs(lim_x))
    num_workers = size - 1
    closed_workers = 0

    print "MASTER: starting with %d workers" % num_workers

    # init_query = np.asarray([[i] for i in np.linspace(0, lim_x[1], init_size)],
    #                         dtype=np.float32) # Uniform sampling
    init_query = np.asarray([[np.random.uniform(0, 1)] for _ in range(init_size)],
                            dtype=np.float32) # Random uniform sampling
    domain = np.asarray([[i] for i in np.linspace(-1, 1, 1000)])

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
    print "Selection size is: " + str(selection_size)

    selection_index = 0
    trainer_dataset_index = 0
    trainer_is_ready = True

    tasks_done = 0
    tasks_total = 50

    # while False:
    while closed_workers < num_workers:
        if selection_index == selection_size:
            optimizer.update()
            selected_points = optimizer.select_multiple()
            selection_size = selected_points.shape[0]
            selection_index = 0
            
        if trainer_is_ready:
            # If trainer is ready, keep shoving data at him, if there is data to be shoved
            print "MASTER: Trainer has been activated"
            additional_dataset = dataset[trainer_dataset_index: -1, :]
            comm.send(additional_dataset, dest=TRAINER, tag=SEND_TRAINER)
            trainser_dataset_index = dataset.shape[0] - 1
            trainer_is_ready = not trainer_is_ready

        if tasks_done == tasks_total:
            print "MASTER: Killing Trainer"
            comm.send("MASTER has fired Trainer", dest=TRAINER, tag=EXIT_TRAINER)

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
                print "MASTER: Killing Worker %2d" % source
                comm.send(None, dest=source, tag=EXIT_WORKER)

        elif tag == WORKER_DONE:
            # If worker is done, tally the total amount of work done. 
            dataset = np.concatenate((dataset, data), axis=0)
            optimizer.update_data(data)
            tasks_done += 1
            string = "MASTER: Number of total tasks: %3d. New data from WORKER %2d is: " % (tasks_done, source)
            print string + str(data)

        elif tag == TRAINER_DONE:
            # If trainer is done, store what trainer did. 
            print "MASTER: Updating feature extractor"
            optimizer.update_feature_extractor(data)
            train_is_ready = not trainer_is_ready

        elif tag == EXIT_WORKER or tag == EXIT_TRAINER:
            # If worker has exited, tally up number of closed workers.
            closed_workers += 1

    t2 = time.time()
    print "MASTER: Total update time is: %3.3f" % (t2-t1)
    # Plot results
    if plot_it:
        plt.gcf().set_size_inches(8, 8)
        true_func = [true_evaluate(domain[i, :], scale)[0, :].tolist() for i in range(domain.shape[0])]
        true_func = np.array(true_func)
        # optimizer.train()
        selected_point = optimizer.select_multiple()[0, :]
        print "MASTER: Final selection: " + str(selected_point)
    
        domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()
        ax = plt.gca()
        plt.plot(true_func[:, :-1], true_func[:, -1:], 'k', 
                 label='True Function',
                 linewidth=3)
        plt.plot(domain, pred, 'c', label='NN-LR Regression', linewidth=3)
        # plt.plot(domain, nn_pred, 'r--', label='NN regression', linewidth=7)
        plt.plot(domain, hi_ci, 'g--', label='Confidence Interval')
        plt.plot(domain, lo_ci, 'g--')
        # plt.plot(domain, ei, 'b--', label='ei')
        # plt.plot(domain, gamma, 'r', label='gamma')
        # plt.plot([selected_point, selected_point], [ax.axis()[2], ax.axis()[3]], 'r--',
        #          label='EI selection')
        plt.plot(dataset[:,:-1], dataset[:, -1:], 'rv', markersize=7.)
        plt.xlabel('Hyperparameter Domain')
        plt.ylabel('Objective Function')
        plt.title("Neural Network Regression")
        plt.legend()
        plt.savefig('figures/test_regression.eps', format='eps', dpi=2000)

        plt.clf()
        plt.gcf().set_size_inches(8, 8)
        plt.plot(domain, ei, 'r', label='Expected Improvement')
        plt.plot(domain, ((hi_ci-pred)/2)**2, 'g', label='Variance')
        plt.xlabel('Hyperparameter Domain')
        plt.ylabel('Expected Improvement')
        plt.title("Selection Criteria")
        plt.legend()
        plt.savefig('figures/test_expected_improvement.eps', format='eps', dpi=2000)

        

def trainer_process():
    import utilities.neural_net as nn
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


def worker_process(lim_x, rank):
    from learning_objective.hidden_function import evaluate
    scale = np.max(np.abs(lim_x))
    while True:
        comm.send("WORKER is ready", dest=0, tag=WORKER_READY)    # tell Master node that I need a new query
        query  = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_WORKER:
            # string = "WORKER %3d: The query is: " % rank
            # print string + str(query)  
            result = evaluate(query, scale)
            comm.send(result, dest=0, tag=WORKER_DONE)

        elif tag == EXIT_WORKER:
            # Worker dies!
            print "WORKER: Worker %2d commiting suicide" % rank
            break

    comm.send(None, dest=0, tag=EXIT_WORKER) # Suicide complete
    
lim_x        = [-6, 6]                                     # x range for univariate data

if rank == MASTER:                         # MASTER NODE
    # Settings
    init_size = 50
    master_process(lim_x, init_size)

elif rank == TRAINER:
    trainer_process()
else:
    worker_process(lim_x, rank)
