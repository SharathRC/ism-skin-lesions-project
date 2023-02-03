import numpy as np 
from add_ones import add_ones
from sigmoid_function import sigmoid

def predict(X, theta):
    X = add_ones(X)
    theta = theta.reshape(-1,1)
    return 1*(sigmoid(X.dot(theta)) > 0.5)