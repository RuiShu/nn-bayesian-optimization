import numpy as np
import time
import scipy.stats as stats
from learning_objective.hidden_function import evaluate, true_evaluate, get_settings
import pyGPs

def contains_row(x, X):
    """ Checks if the row x is contained in matrix X
    """
    for i in range(X.shape[0]):
        if all(X[i,:] == x):
            return True

    return False

def select(dataset, pred_y, sigma2_pred):
    """ Identify multiple points. 
    """
    
    # Rank order by expected improvement
    train_Y    = dataset[:, -1:]
    prediction = pred_y
    sig = sigma2_pred**0.5

    gamma = (prediction - np.max(train_Y)) / sig
    ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))

    if np.max(ei) <= 0:
        select_index = np.argmax(sig)
    else:
        select_index = np.argmax(ei)

    return np.atleast_2d(domain[select_index, :])


if __name__ == "__main__":
    print_statements = False

    # Open file to write times for comparison
    file_record = open("data/gp_time_data.csv", "a")

    # Get settings relevant to the hidden function being used
    lim_domain, init_size, additional_query_size, init_query, domain, selection_size = get_settings()

    # Construct the dataset
    dataset = evaluate(init_query[0,:], lim_domain)

    print "Randomly query a set of initial points... ",

    for query in init_query[1:,:]:
        dataset = np.concatenate((dataset, evaluate(query, lim_domain)), axis=0)

    print "Complete initial dataset acquired"
    
    # Begin sequential optimization using Gaussian process based query system
    model = pyGPs.GPR() 
    model.getPosterior(dataset[:,:-1], dataset[:,-1:])
    model.optimize(dataset[:,:-1], dataset[:,-1:])    
    model.predict(domain)        
    y_pred = model.ym
    sigma2_pred = model.ys2
    query = select(dataset, y_pred, sigma2_pred)

    print "Performing optimization... "

    for i in range(additional_query_size):
        t0 = time.time()
        new_data = evaluate(query, lim_domain)
        dataset = np.concatenate((dataset, new_data), axis=0)

        if print_statements:
            string1 = "Tasks done: %3d. " % (i+1)
            string2 = "New data added to dataset: " + str(new_data)
            print string1 + string2
            
        else:
            if (i+1) % (additional_query_size/10) == 0:
                print "%.3f completion..." % ((i+1.)/additional_query_size)

        model.getPosterior(dataset[:,:-1], dataset[:,-1:])

        try:
            model.optimize(dataset[:,:-1], dataset[:,-1:])    
        except:
            pass

        model.predict(domain)    
        y_pred = model.ym
        sigma2_pred = model.ys2
        query = select(dataset, y_pred, sigma2_pred)

        info = "%.3f," % (time.time()-t0)
        file_record.write(info)

    file_record.write("NA\n")
    file_record.close()
        
    print "Sequential gp optimization task complete."
    print "Best evaluated point is:"
    print dataset[np.argmax(dataset[:, -1]), :]
    print "Predicted best point is:"
    index = np.argmax(y_pred[:, 0])
    print np.concatenate((np.atleast_2d(domain[index, :]), np.atleast_2d(y_pred[index, 0])), axis=1)[0, :]
