import numpy as np
import time
import scipy.stats as stats
from sklearn import gaussian_process
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
        print "optimizer.py: Pure exploration"
    else:
        select_index = np.argmax(ei)
        print "optimizer.py: All expected"

    return np.atleast_2d(domain[select_index, :])


if __name__ == "__main__":
    # Open file to write times for comparison
    f = open("data/crap_gp_350obs_time_data.csv", "a")

    # Get settings relevant to the hidden function being used
    lim_domain, init_size, additional_query_size, init_query, domain, selection_size = get_settings()

    # Construct the dataset
    dataset = evaluate(init_query[0,:], lim_domain)

    for query in init_query[1:,:]:
        dataset = np.concatenate((dataset, evaluate(query, lim_domain)), axis=0)

    print "Complete initial dataset acquired"
    print dataset
    
    # Begin sequential optimization using Gaussian process based query system
    model = pyGPs.GPR() 
    model.getPosterior(dataset[:,:-1], dataset[:,-1:])
    model.optimize(dataset[:,:-1], dataset[:,-1:])    
    model.predict(domain)        
    y_pred = model.ym
    sigma2_pred = model.ys2
    query = select(dataset, y_pred, sigma2_pred)

    for i in range(additional_query_size):
        t0 = time.time()
        new_data = evaluate(query, lim_domain)
        dataset = np.concatenate((dataset, new_data), axis=0)
        string1 = "Tasks done: %3d. " % (i+1)
        string2 = "New data added to dataset: " + str(new_data)
        print string1 + string2

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
        f.write(info)

    # f.write("NA\n")
    f.close()
        
    print "Sequential gp optimization task complete."
    print "Best evaluated point is:"
    print dataset[np.argmax(dataset[:, -1]), :]
    print "Predicted best point is:"
    index = np.argmax(y_pred[:, 0])
    print np.concatenate((np.atleast_2d(domain[index, :]), np.atleast_2d(y_pred[index, 0])), axis=1)[0, :]

    # for i in range(additional_query_size):
    #     gp.fit(dataset[:,:-1], dataset[:,-1:])
    #     y_pred, sigma2_pred = gp.predict(domain, eval_MSE=True)
    #     query = select(dataset, y_pred, sigma2_pred)
