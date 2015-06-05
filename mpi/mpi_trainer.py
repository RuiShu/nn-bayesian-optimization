"""
@Author: Rui Shu
@Date: 4/11/15

Trainer -- in charge of handling the neural network training. 
"""
from mpi_definitions import *
from theano_definitions import *

def trainer_process(print_statements):
    import utilities.neural_net as nn
    nobs = 0
    dataset = None
    untrained_data_count = 0

    while True:
        new_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_TRAINER:
            if print_statements:
                print "TRAINER: Received from master, starting new neural net"

            if dataset == None:
                dataset = new_data

            else:
                dataset = np.concatenate((dataset, new_data), axis=0)

            nobs = dataset.shape[0]
            architecture = (dataset.shape[1] - 1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
            neural_net = nn.NeuralNet(architecture, dataset)
            neural_net.train()
            W, B = neural_net.extract_params()

            if print_statements:
                print "TRAINER: Sending back neural net to master"

            comm.send((W, B, architecture), dest=0, tag=TRAINER_DONE)
            
        elif tag == EXIT_TRAINER:
            print "TRAINER: Exiting"
            break

    comm.send(None, dest=0, tag=EXIT_TRAINER) # Suicide complete


