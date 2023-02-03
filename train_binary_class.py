import numpy as np
import csv
import pandas as pd 
from scipy import optimize as opt
from read_data import read_data
from add_ones import add_ones
from predict import predict
from f1score_calc import f1score_calc

import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

from costFuncLogReg import cost_func_log_reg, grad_log_reg
from featureNormalizer import featureNormalizer

import torch
import torch.nn as nn

kernel_type = 'rbf' #'poly' 'linear'  'rbf'

def randomize(X, Y):
    m,n = X.shape
    Xrand = []
    Yrand = []
    i = np.arange(m)
    np.random.shuffle(i)
    for index in i:
        Xrand = np.append(Xrand, X[index][:])

        Yrand = np.append(Yrand, Y[index][:])

    Xrand = Xrand.reshape(-1,n)
    Yrand = Yrand.reshape(-1,1)

    return Xrand, Yrand

def seperate_data(X,Y):
    m,n = X.shape
    Xpos = []
    Ypos = []
    Xneg = []
    Yneg = []
    for i in range(m):
        if Y[i] == 1:
            Xpos = np.append(Xpos, X[i][:])
        else:
            Xneg = np.append(Xneg, X[i][:])
    
    Xpos = Xpos.reshape(-1,n)
    Xneg = Xneg.reshape(-1,n)
    Ypos = np.ones(( Xpos.shape[0],1 ))
    Yneg = np.zeros(( Xneg.shape[0],1 ))

    return Xpos, Ypos, Xneg, Yneg

def test():
    path_train_X = '../data/TRAINING_DATA/features_25hist_contour_otsu_12texture.csv' #train
    path_train_Y = '../data/TRAINING_DATA/train_data.csv'
    path_test_X = '../data/TEST_DATA/features_25hist_contour_otsu_12texture_cont_2.csv' #test
    TESTING_SIZE = 100
    
    
    ##################################################################
    #Training data extraction
    X, Y = read_data(1, path_train_X, True)
    num_samples, num_features = X.shape
    print('X shape: {}'.format(X.shape))
    print('Y shape: {}'.format(Y.shape))
    
    Xpos, Ypos, Xneg, Yneg = seperate_data(X,Y)
    Xpos_rand, Ypos_rand = randomize(Xpos, Ypos)
    Xneg_rand, Yneg_rand = randomize(Xneg, Yneg)
    NUM_POS = Xpos.shape[0]
    
    print('Xpos shape: {}, Xneg shape: {}'.format(Xpos.shape, Xneg.shape))

    Xrand = np.concatenate(( Xpos_rand[ 0:int(NUM_POS/1.5) ][:], Xneg_rand[ 0:int(NUM_POS/1.5) ][:] ))
    Yrand = np.concatenate(( Ypos_rand[ 0:int(NUM_POS/1.5) ][:], Yneg_rand[ 0:int(NUM_POS/1.5) ][:] ))
    Xrand, Yrand = randomize(Xrand,Yrand)

    X_normalized = featureNormalizer(Xrand)
    X_train = X_normalized
    Y_train = np.ravel(Yrand)
    print('X_train shape: {}'.format(X_train.shape))

    ###################################################################
    #test data
    limit = int(NUM_POS/1.5) + int(TESTING_SIZE/2)
    Xrand = np.concatenate(( Xpos_rand[ int(NUM_POS/1.5):limit ][:], Xneg_rand[ int(NUM_POS/1.5):limit ][:] ))
    Yrand = np.concatenate(( Ypos_rand[ int(NUM_POS/1.5):limit ][:], Yneg_rand[ int(NUM_POS/1.5):limit ][:] ))
    Xrand, Yrand = randomize(Xrand,Yrand)

    X_normalized = featureNormalizer(Xrand)
    X_test = X_normalized
    Y_test = np.ravel(Yrand)
    print('X_test shape: {}'.format(X_test.shape))

    ###################################################################
    #final test data
    #X = read_data(1, path_test_X, False)
    X_full = np.genfromtxt (path_test_X, delimiter=",")
    X = X_full[:,1:]
    # Y = np.genfromtxt (path_train_Y, delimiter=",")
    # Y = Y[:,8]
    #print('X_test_final shape: {}'.format(X.shape))
    X_normalized = featureNormalizer(X)
    X_test_final = X_normalized
    print('X_test_final shape: {}'.format(X_test_final.shape))
    
    ###################################################################
    #training

    #Result = opt.fmin_bfgs(f = cost_func_log_reg, x0 = init_theta, args = (X_normalized, Yrand, L), maxiter = 400, fprime = grad_log_reg)
    
    # clf = svm.SVC(kernel = kernel_type, C=20)
    # clf.fit(X_train, Y_train)
    # clf_predictions = clf.predict(X_test)
    # print("Accuracy (w/o grid search): {}%".format(clf.score(X_test, Y_test) * 100 ))
    # F1 = f1score_calc(Y_test, clf_predictions)
    # print("F1 score (w/o grid search): {}".format(F1))
    # clf_predictions_final = clf.predict(X_test_final)
    # # print('weights: ')
    # # print(clf.coef_)

    # print(X_full[:,0].shape)
    # print(clf_predictions_final.shape)

    # Grid Search
    # Parameter Grid
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(kernel = kernel_type), param_grid, verbose=1)
    
    # Train the classifier
    clf_grid.fit(X_train, Y_train)
    
    # clf = grid.best_estimator_()
    print("Best Parameters:\n", clf_grid.best_params_)
    print("Best Estimators:\n", clf_grid.best_estimator_)

    clf_predictions = clf_grid.predict(X_test)
    print("Accuracy (w grid search): {}%".format(clf_grid.score(X_test, Y_test) * 100 ))
    F1 = f1score_calc(Y_test, clf_predictions)
    print("F1 score (w grid search): {}".format(F1))
    clf_predictions_final = clf_grid.predict(X_test_final)

    print(X_full[:,0].shape)
    print(clf_predictions_final.shape)
    # params = clf_grid.get_params(True)
    # print('weights: ')
    # print(clf_grid.coef_)
    
    ###################################################################
    # Neural network PYTORCH (2nd model)

    n_in, n_h, n_out, batch_size = num_features, 80, 1, num_samples 
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    #x = torch.randn(batch_size, n_in)
    #y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    x = torch.from_numpy(X_train)
    y = torch.from_numpy(Y_train)
    #y = torch.tensor(Y)

    #print(x)
    model = nn.Sequential(nn.Linear(n_in, n_h), nn.ReLU(), nn.Linear(n_h, n_out), nn.Sigmoid())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(50):
        # Forward Propagation
        y_pred = model(x)
        # Compute and print loss
        loss = criterion(y_pred, y)
        #print('epoch: ', epoch,' loss: ', loss.item())
        # Zero the gradients
        optimizer.zero_grad()
        
        # perform a backward pass (backpropagation)
        loss.backward()
        
        # Update the parameters
        optimizer.step()
    
    X_test = X_test.astype(np.float32)
    x_test = torch.from_numpy(X_test)
    y_pred = model(x_test)
    pyT_predictions = 1*(y_pred.detach().numpy() >= 0.5)
    #pyT_predictions = pyT_predictions >= 0.5
    F1 = f1score_calc(Y_test, pyT_predictions)
    print("F1 score (w pytorch): {}".format(F1))

    ###################################################################
    #write results to file

    names = X_full[:,0].reshape(-1,1)
    names = names.astype(int)
    names = names.astype(str)
    names = np.core.defchararray.add('ISIC00', names)
    labels_final = clf_predictions_final.reshape(-1,1)
    labels_final = labels_final.astype(int)
    results_to_write = np.concatenate((names, labels_final), axis = 1)
    print(results_to_write.shape)
    df = pd.DataFrame(results_to_write)
    df.to_csv("result_path.csv", header = None, index=None)
    
    
    ###################################################################
    # # Create a linear SVM classifier with C = 1
    # clf = svm.SVC(kernel='linear', C=1)
    # # Create SVM classifier based on RBF kernel. 
    # clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)
    


test()

