class Master(object):
    def __init__(self, size):
        """Initialization of Master object
        
        Keyword arguments:
        architecture -- a tuple containing the number of nodes in each layer
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        
        self.tasks_done = 0
        self.tasks_total = 4
        self.num_workers = size - 1
        self.closed_workers = 0
        self.trainer_is_ready = True

    def push_to_trainer():
        pass
    
if __name__ == "__main__":
    master = Master(3)
