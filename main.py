import numpy as np
import theanets
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

def train_neural_net(dataset, architecture):
    cut = int(0.9 * len(dataset))  # select 90% of data for training, 10% for validation
    idx = range(len(dataset))
    np.random.shuffle(idx)
    
    train = idx[:cut]
    train_set = [dataset[train, :1], dataset[train, 1:]]
    valid = idx[cut:]
    valid_set = [dataset[valid, :1], dataset[valid, 1:]]
    
    e = theanets.Experiment(theanets.feedforward.Regressor,
                            layers=architecture,
                            optimize='sgd',
                            activation='tanh', 
                            output_activation='linear',
                            learning_rate=0.01)
    
    e.run(train_set, valid_set)
    return e

def extract_params(e, architecture):
    # Extract parameters
    W = {}
    B = {}
    for i in range(len(architecture)-2):
        W[i] = e.network.params[2*i].get_value()
        B[i] = np.reshape(e.network.params[2*i+1].get_value(), (1, architecture[i+1]))
        
    return W, B

def extract_features(test, W, B):
    # Feedforward into custom neural net
    X = []

    for test_val in test:
        L = np.tanh(np.dot(test_val, W[0]) + B[0])
        
        for i in range(1, len(architecture)-2):
            L = np.tanh(np.dot(L, W[i]) + B[i])

        X.extend(L.tolist())

    X = np.asarray(X)
    X = sm.add_constant(X)

    return X

def lm_predict(train_X, train_Y, test_X):
    # # Training and prediction
    # XX_inv = np.linalg.inv(np.dot(train_X.T, train_X))
    # print XX_inv
    # Why is lstsq not the same as inv. WHYYY
    XX_inv,_,_,_ = np.linalg.lstsq(np.dot(train_X.T, train_X), np.identity(train_X.shape[1]))

    beta = np.dot(np.dot(XX_inv, train_X.T), train_Y)
    train_pred = np.dot(train_X, beta)

    # Confidence interval
    sig = (np.linalg.norm(train_Y-train_pred)**2/(train_X.shape[0]-train_X.shape[1]+1))**0.5

    s = []
    for row in range(test_X.shape[0]):
        x = test_X[[row], :]
        s.append(sig*(1 + np.dot(np.dot(x, XX_inv), x.T))**0.5)

    s = np.reshape(np.asarray(s), (test_X.shape[0], 1))

    test_pred = np.dot(test_X, beta)
    hi_ci = test_pred + 2*s
    lo_ci = test_pred - 2*s

    return test_pred, hi_ci, lo_ci

def select_ei_index(train_Y, prediction, hi_ci):
    sig = (hi_ci - prediction)/2
    # gamma = (min(train_Y) - prediction)/sig # finding min
    gamma = -(min(train_Y) - prediction)/sig # finding max
    ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))
    
    return np.argmax(ei)

# if __name__ == "__main__":
if __name__ == "__main2__":
    g = lambda x: 2*x+1 + np.random.randn()/2
    
    train_X = np.sort(np.asarray([[np.random.randn()/2] for _ in range(10)]), axis=0)
    train_Y = (2*train_X + 1) + np.random.randn(train_X.shape[0], 1)
    test_X = np.asarray([[i] for i in np.linspace(-5, 5, 100)])

    # Add bias
    X1 = sm.add_constant(train_X) 
    X2 = sm.add_constant(test_X) 

    pred, hi_ci, lo_ci = lm_predict(X1, train_Y, X2)

    plt.plot(train_X, train_Y, 'ro', label="true")
    plt.plot(test_X, pred, label="linear")
    plt.plot(test_X, hi_ci, label="ci")
    plt.plot(test_X, ci2, label="ci")
    plt.xlabel('x label'); plt.ylabel('y label'); plt.title("Simple Plot")
    plt.legend()
    plt.show()

if __name__ == "__main__":
# if __name__ == "__main2__":
    # Settings
    lim_x        = [-1, 1]                                                  # x range for univariate data
    nobs         = 100                                                       # number of observed data
    architecture = (1, 50, 50, nobs-2 if nobs < 50 else 50, 1)                                           # Define NN layer architecture
    noiseless_g  = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2             
    g            = lambda x: np.exp(-x)*np.sin(10*x)*x-4*x**2 + np.random.randn()/10 # Define the hidden function
    # g = lambda x: 2*x**3-x**2+3 + np.random.randn()/10
    # noiseless_g = lambda x: 2*x**3-x**2+3
    
    # Define the dataset
    dataset_X = np.asarray([[i] for i in np.linspace(lim_x[0], lim_x[1], nobs)], dtype=np.float32) # Uniform sampling
    # dataset_X2 = np.asarray([[-0.5] for _ in np.linspace(lim_x[0], lim_x[1], 100)], dtype=np.float32) # Uniform sampling
    # dataset_X = np.concatenate((dataset_X, dataset_X2), axis=0)
    # dataset_X = np.sort(np.asarray([[np.random.randn()/2] for _ in range(nobs)]), axis=0) # Random sampling
    dataset_Y = np.asarray([[g(dataset_X[i, :])[0]] for i in range(dataset_X.shape[0])])
    dataset = np.concatenate((dataset_X, dataset_Y), axis=1)

    # Train the neural net
    e = train_neural_net(dataset, architecture) 

    # Extract parameters
    W, B = extract_params(e, architecture)

    # Feature extraction
    train_Y = dataset[:, [1]]
    train_X = dataset[:, [0]]
    test_X = np.array([[i] for i in np.linspace(lim_x[0], lim_x[1], 50)], dtype=np.float32)

    train_features = extract_features(train_X, W, B)
    test_features = extract_features(test_X, W, B)

    # Predictions
    # my linear regression
    lm_prediction, hi_ci, lo_ci = lm_predict(train_features, train_Y, test_features)
    # NEURAL NETWORK prediction
    nn_prediction = e.network(test_X)
    
    index = select_ei_index(train_Y, lm_prediction, hi_ci)
    print "Max expected imrpovement at x = " + str(test_X[index, :])

    ax = plt.gca()
    plt.plot(test_X, nn_prediction, label='NN')
    plt.plot(test_X, lm_prediction, 'c--', label='linear reg', linewidth=7)
    plt.plot(test_X, hi_ci, 'g--', label='ci')
    plt.plot(test_X, lo_ci, 'g--')
    true_func = np.asarray([[i, noiseless_g(i)] for i in np.linspace(lim_x[0], lim_x[1], 100)], dtype=np.float32)
    plt.plot(true_func[:, 0], true_func[:, 1], 'k', label='true', linewidth=4) # true plot
    plt.plot(train_X, train_Y, 'rv', label='training', markersize=7.)
    plt.plot([test_X[index, :], test_X[index, :]], [ax.axis()[2], ax.axis()[3]], 'r--',
             label='EI selection')
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.show()
