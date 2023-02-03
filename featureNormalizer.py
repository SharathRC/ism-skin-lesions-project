import numpy as np

def featureNormalizer(X):
    mean = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mean)/sigma
    return X