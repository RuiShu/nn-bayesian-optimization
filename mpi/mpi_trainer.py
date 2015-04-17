"""
@Author: Rui Shu
@Date: 4/11/15

Trainer -- in charge of handling the neural network training. 
"""
from mpi_definitions import *

def trainer_process():
    import utilities.neural_net as nn
    nobs = 0
    dataset = None
    untrained_data_count = 0

    while True:
        new_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == SEND_TRAINER:
            if dataset == None:
                dataset = new_data
            else:
                dataset = np.concatenate(dataset, new_data)

            nobs = dataset.shape[0]
            print "TRAINER: Received from master"
            print "TRAINER: Starting new feature extractor"
            architecture = (1, 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
            feature_extractor = nn.NeuralNet(architecture, dataset)
            feature_extractor.train()
            test = feature_extractor.extract_params()
            print "TRAINER: Sending back to master"
            comm.send(test, dest=0, tag=TRAINER_DONE)
            
        elif tag == EXIT_TRAINER:
            print "TRAINER: Commiting suicide"
            break

    comm.send(None, dest=0, tag=EXIT_TRAINER) # Suicide complete


