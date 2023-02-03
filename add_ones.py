import numpy as np 

def add_ones(X):
    m, n = X.shape
    X = np.concatenate( ( np.ones((m,1)), X ), axis = 1)
    return X