import numpy as np
from sigmoid_function import sigmoid


def cost_func_log_reg(theta, X, Y, L):
    m, n = X.shape
    theta = theta.reshape(n,1)
    Y = Y.reshape((m,1))
    y = Y
    h = sigmoid(np.dot(X, theta))
    #print(h)
    theta_mod = np.concatenate(( np.zeros((1, np.size(theta, 1))), theta[1:, :] ))
    #J = (1/m)*np.sum(-Y*np.log(h) - (1-Y)*np.log(1-h)) + (0.5*L/m)*np.sum(theta_mod**2)
    
    J = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + (0.5*L/m)*np.sum(theta_mod**2)
    
    #print(np.log(1 - h))
    return J


def grad_log_reg(theta, X, Y, L):
    m, n = X.shape
    theta = theta.reshape(n,1)
    Y = Y.reshape((m,1))
    h = sigmoid(np.dot(X, theta))
    theta_mod = np.concatenate((np.zeros((1, np.size(theta, 1))), theta[1:, :]))
    grad = (1/m)*np.array([np.sum((h-Y)*X, 0)]) + (L/m)*np.sum(theta_mod,1)
    grad = np.asarray(grad)
    grad = grad.flatten()
    return grad
