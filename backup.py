"""@Author: Rui Shu
@Date: 4/11/15

Finds the global maxima of a costly function in a parallelized setting. Runs
optimizer.py in parallel with with several worker nodes that evaluates the costly
function in batch. 

Run as: mpiexec -np N python parallel_optimizer.py
where N is the number of available processes

Framework: 
Master -- handles the Optimizer object (which takes prior data,
interpolates based on a neural network-linear regression model, and selects the
next set of points to query). Tells worker nodes which points to query.

Worker -- compute the costly function. Returns function evaluation.

Trainer -- in charge of handling the neural network training. 
"""

from mpi4py import MPI
import numpy as np

# Mimic enumeration. To be used for MPI tags.
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
  
# Define MPI message tags
tags = enum('SEND_WORKER',  # Send by Master: gives Worker some work
            'SEND_TRAINER', # Send by Master: gives Trainer more data
            'EXIT_WORKER',  # Send by Master/Worker: tells Worker to quit
            'EXIT_TRAINER', # Send by Master/Trainer: tells Trainer to quit
            'WORKER_READY', # Send by Worker: tells Master to give work
            'WORKER_DONE',  # Send by Worker: tells Master the work is done
            'TRAINER_DONE') # Send by Trainer: gives Master new neural network

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

if rank == 0:                         # MASTER NODE
    num_workers = size - 1
    closed_workers = 0

    print("Master starting with %d workers" % num_workers)

    while closed_workers < num_workers:

        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status) # receive worker info
        source = status.Get_source() 
        tag = status.Get_tag()          # check what worker's request is

        if tag == tags.WORKER_READY:
            comm.send("Send worker some query", dest=source, tag=tags.SEND_WORKER)

        elif tag == tags.WORKER_DONE:
            print data

        elif tag == tags.TRAINER_DONE:
            print data

        elif tag == tags.EXIT_WORKER:
            closed_workers += 1

        elif tag == tags.EXIT_TRAINER:
            closed_workers += 1

        else:
            print "Should this really be happening?"

    print "Master is done."

    #     if tag == tags.NEEDCONTROL:     # worker wants to know what the input sentence is
    #         comm.send(control, dest=source, tag=tags.GIVECONTROL)
            
    #     elif tag == tags.READY:         # worker is ready
    #         if task_end < tasks:        # there're still tasks left, so send it a task 
    #             print "Rank 0 sent: %d to %d" % (task_start, task_end)
    #             comm.send([task_start, task_end], dest=source, tag=tags.START) # task indexed [a,b] inclusive
    #             task_start += task_step
    #             task_end += task_step

    #         elif task_end > tasks and task_start <= tasks: # final left over task
    #             print "Rank 0 sent: %d to %d" % (task_start, tasks)
    #             comm.send([task_start, tasks], dest=source, tag=tags.START)
    #             task_start = tasks + 1

    #         else:                              # no more tasks left, close worker
    #             comm.send(None, dest=source, tag=tags.EXIT)
                
    #     elif tag == tags.DONE:                 # worker is done.
    #         if abs(data[1]-1) < abs(1-maxval): # check if returned sentence has better string kern val
    #             right_sentence = data[0]
    #             maxval = data[1]

    #         completed += 1                     # increment count of completed partitions.
    #         print "%.2f complete." % (1.*completed/math.ceil(1.*tasks/task_step))
            
    #     elif tag == tags.EXIT:
    #         closed_workers += 1
  
    # print "I think the right name is: %s, match: %.5f" % (right_sentence, maxval)
    # t1 = time.time()
    # print "Time taken: %.5f" % (t1-t0)

elif rank > 0 and rank < (size - 1):

    while True:
        comm.send(None, dest=0, tag=tags.NEED_QUERY)    # tell Master node that I need a new query
        query  = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.SEND_QUERY:
            print query
            # Does its thing
            comm.send(None, dest=0, tag=tags.SEND_EVAL)

        elif tag == tags.EXIT_QUERY:
            print "Commiting suicide"
            break

        else:
            print "This really shouldn't be happening"

    comm.send(None, dest=0, tag=tags.EXIT_QUERY) # Suicide complete

else:

    while True:
        new_data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.SEND_DATA:
            print new_data
            # Does its thing
            comm.send(None, dest=0, tag=tags.SEND_NEURAL)
        elif tag == tags.EXIT_NN:
            break
        else:
            print "This should not be happending"

    comm.send(None, dest=0, tag=tags.EXIT_NN)
    
    # while True:
    #     comm.send(None, dest=0, tag=tags.READY)        # tell Maste node that I'm ready for work
    #     task = comm.recv(source=0, tag=MPI.ANY_SOURCE, status=status)
    #     tag = status.Get_tag()

    #     if tag == tags.START:                          # master node gave me work
    #         if (task[1] + 1 - task[0]) < len(control): # cannot construct substring in story_text of len(control)
    #             comm.send(["", 0], dest=0, tag=tags.DONE)

    #         else:
    #             right_sentence = ""
    #             maxval = 0

    #             for i in range(task[0],task[1]+1):                                    # iterate through [task_start, task_end] 
    #                 if (i+len(control)) > len(story_text):                            # unable to construct story_text[i:(i+len(control))]
    #                     break

    #                 sentence = story_text[i:(i+len(control))]                         # generate contiguous substring
    #                 vector2 = story.kernel_vector(sentence.lower(), gram_set)         # construct kernel vector for string
    #                 checkval = story.evaluate(vector1, vector2, gram_set)/norm_factor # compare against control
                    
    #                 if abs(checkval-1) < abs(maxval-1):
    #                     maxval = checkval
    #                     right_sentence = sentence

    #             comm.send([right_sentence, maxval], dest=0, tag=tags.DONE)

    #     elif tag == tags.EXIT:
    #         break
  
