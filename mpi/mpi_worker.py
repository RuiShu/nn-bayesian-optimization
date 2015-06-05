"""
@Author: Rui Shu
@Date: 4/11/15

Worker -- compute the costly function. Returns function evaluation.
"""

from mpi_definitions import *

def worker_process(rank):
    from learning_objective.hidden_function import evaluate, get_settings

    lim_domain = get_settings(lim_domain_only=True)

    while True:
        comm.send("WORKER is ready", dest=0, tag=WORKER_READY)    # tell Master node that I need a new query
        query  = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == SEND_WORKER:
            # string = "WORKER %3d: The query is: " % rank
            # print string + str(query)  
            result = evaluate(query, lim_domain)
            comm.send(result, dest=0, tag=WORKER_DONE)

        elif tag == EXIT_WORKER:
            # Worker dies!
            print "WORKER: Worker %2d exiting" % rank
            break

    comm.send(None, dest=0, tag=EXIT_WORKER) # Suicide complete
    
