import numpy as np

def centroid(X):
    m,n = X.shape
    c = np.sum(X,0)/m
    return c