import numpy as np
import csv
import pandas as pd 

from scipy import optimize as opt
from read_data_mult_class import read_data_multi
from add_ones import add_ones
from predict import predict
from f1score_calc import f1score_calc

import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

from costFuncLogReg import cost_func_log_reg, grad_log_reg
from featureNormalizer import featureNormalizer

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

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

def calc_accuracy(Y_test, pyT_predictions):
    confusion_mat = confusion_matrix(Y_test, pyT_predictions)
    print(confusion_mat)
    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)  
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)
    print(' FP: {} \n FN: {} \n TP: {} \n TN: {}'.format(FP, FN, TP, TN))
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Training accuracy: {}'.format(ACC))

def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)

def test():
    #path_train_X = '../data/TRAINING_DATA/features_25hist_contour_otsu_12texture.csv' #train
    path_train_X = '../data/TRAINING_DATA/features_25hist_contour_otsu_12texture_cont_hair_removed_ordered.csv' #train
    path_train_Y = '../data/TRAINING_DATA/train_data.csv'
    #path_test_X = '../data/TEST_DATA/features_25hist_contour_otsu_12texture_cont_2.csv' #test
    path_test_X = '../data/TEST_DATA/features_25hist_contour_otsu_12texture_cont_hair_removed_test_ordered.csv' #test
    
    TRAINING_SIZE = 8000
    TESTING_SIZE = 100
    
    
    ##################################################################
    #Training data extraction
    X, Y = read_data_multi(1, path_train_X, True)
    #print(Y)
    num_samples, num_features = X.shape
    print('X shape: {}'.format(X.shape))
    print('Y shape: {}'.format(Y.shape))
    #X1, X2, X3, X4, X5, X6 = []
    X_rearranged = [[],[],[],[],[],[], []]
    X, Y = randomize(X,Y)
    nlabels = 7
    labels = [1, 2, 3, 4, 5, 6, 7]
    
    count = 0
    for data in Y:
        count+=1
        for i in labels:
            if data == i:
                X_rearranged[i-1] = np.append(X_rearranged[i-1], X[count-1,:])
                break
    
    print('count: {}'.format(count))
    X_train = [] 
    Y_train = [] 
    X_test = [] 
    Y_test = []
    total_samples = 0
    
    #split_factor = [8, 60, 2.8, 2, 5, 1.1, 1.1]
    split_factor = [2, 10, 1.5, 1.5, 1.5, 1.1, 1.1]
    
    for i in labels:       
        n = len(X_rearranged[i-1])       
        samples = n/num_features        
        size = int(samples/split_factor[i-1])
        #size = 90
        print('num_samples in class {}: {} | chosen for training: {}'.format(i, samples, size))
        total_samples+= samples
        
        X_train = np.append(X_train, X_rearranged[i-1][ 0:(size*num_features) ] )
        Y_train = np.append(Y_train, i * np.ones(( size,1 )) )
        
        X_test = np.append(X_test, X_rearranged[i-1][ (size*num_features): ] )
        Y_test = np.append(Y_test, i * np.ones(( int(samples-size),1 )) )

    print('total samples: {}'.format(total_samples))
    print(X_train.shape)
    
    X_train = X_train.reshape(-1,num_features)
    Y_train = Y_train.reshape(-1,1)
    X_test = X_test.reshape(-1,num_features)
    Y_test = Y_test.reshape(-1,1)
    Xrand, Yrand = randomize(X_train, Y_train)

    X_normalized = featureNormalizer(Xrand)
    X_train = X_normalized
    m,n = X_train.shape
    Y_train = np.ravel(Yrand)
    Y_train_pyT = np.zeros((m,nlabels))
    
    count = 0
    for label in Y_train:
        Y_train_pyT[count][int(label)-1] = 1
        count+=1
    
    print('X_train shape: {}'.format(X_train.shape))
    print('Y_train shape: {}'.format(Y_train.shape))
    print('Y_train_pyT shape: {}'.format(Y_train_pyT.shape))
    print('_______________________________')

    ###################################################################
    #test data
    
    Xrand, Yrand = randomize(X_test, Y_test)

    X_normalized = featureNormalizer(Xrand)
    X_test = X_normalized
    m,n = X_test.shape
    Y_test = np.ravel(Yrand)
    Y_test_pyT = np.zeros((m,nlabels))
    
    count = 0
    for label in Y_test:
        Y_test_pyT[count][int(label)-1] = 1
        count+=1
    
    print('X_test shape: {}'.format(X_test.shape))
    print('Y_test shape: {}'.format(Y_test.shape))
    print('Y_test_pyT shape: {}'.format(Y_test_pyT.shape))
    print('_______________________________')
    
    ###################################################################
    #final test data
    X_full = np.genfromtxt (path_test_X, delimiter=",")
    X_test_final = X_full[:,1:]
    X_normalized = featureNormalizer(X_test_final)
    X_test_final = X_normalized
    print('X_test_final shape: {}'.format(X_test_final.shape))

    ###################################################################
    #multi-class training
    '''
    #Result = opt.fmin_bfgs(f = cost_func_log_reg, x0 = init_theta, args = (X_normalized, Yrand, L), maxiter = 400, fprime = grad_log_reg)
    
    # clf = svm.SVC(kernel = kernel_type, C=20)
    # clf.fit(X_train, Y_train)
    # clf_predictions = clf.predict(X_test)
    # print("Accuracy: {}%".format(clf.score(X_test, Y_test) * 100 ))
    # F1 = f1score_calc(Y_test, clf_predictions)
    # print("F1 score: {}".format(F1))
    # clf_predictions_final = clf.predict(X_test_final)

    # print(X_full[:,0].shape)
    # print(clf_predictions_final.shape)
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(80), random_state=1) #lbfgs
    clf.fit(X_train, Y_train)
    
    clf_predictions = clf.predict(X_test)
    #print(Y_train)
    #print(clf_predictions)
    score = fbeta_score(Y_test, clf_predictions, average='weighted', beta=0.5) #None, binary (default), micro, macro, samples, weighted
    print("Accuracy: {}%".format( score * 100 ))
    print("Accuracy: {}%".format(clf.score(X_test, Y_test) * 100 ))
    
    clf_predictions_final = clf.predict(X_test_final)
    '''
    ###################################################################
    # Neural network PYTORCH

    n_in, n_h1, n_h2, n_h3, n_out, batch_size, epochs = num_features, 75, 45, 15, nlabels, num_samples, 30
    
    X_train = X_train.astype(np.float32)
    Y_train_pyT = Y_train_pyT.astype(np.float32)
    #x = torch.randn(batch_size, n_in)
    #y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    x = torch.from_numpy(X_train)
    y = torch.from_numpy(Y_train_pyT)
    
    model = nn.Sequential(nn.Linear(n_in, n_h1), nn.ReLU(),  nn.Linear(n_h1, n_out), nn.Sigmoid()) # LogSigmoid
    #model = nn.Sequential(nn.Linear(n_in, n_h1), nn.ReLU6(), nn.Linear(n_h1, n_h2), nn.ReLU6(),  nn.Linear(n_h2, n_out), nn.Softmax()) #nn.Sigmoid() nn.Softmax()
    #model = nn.Sequential(nn.Linear(n_in, n_h1), nn.ReLU(), nn.Linear(n_h1, n_h2), nn.ReLU(), nn.Linear(n_h2, n_h3), nn.ReLU(),  nn.Linear(n_h3, n_out), nn.Sigmoid()) #nn.Sigmoid() nn.Softmax()
    #model = nn.Sequential(nn.Linear(n_in, n_h1), nn.ReLU6(),  nn.Linear(n_h1, n_out), nn.Sigmoid())
    #model = nn.Sequential(nn.Linear(n_in, n_h1), nn.ReLU(),  nn.Linear(n_h1, n_out), nn.Softmax())
    
    criterion = nn.MultiLabelSoftMarginLoss() #CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #SGD
    print(model)
    
    for epoch in range(epochs):
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
    pyT_predictions = y_pred.detach().numpy()
    pyT_predictions = (np.argmax(pyT_predictions,1) + 1)
    
    print('predictions: {}'.format(pyT_predictions))
    print('groung truth: {}'.format(Y_test))

    #pyT_predictions = pyT_predictions >= 0.5
    score = fbeta_score(Y_test, pyT_predictions, average='weighted', beta=0.5) #None, binary (default), micro, macro, samples, weighted
    #print("Accuracy (from pyTorch): {}%".format( score * 100 ))
    print(score)
    calc_accuracy(Y_test, pyT_predictions)
    
    ###################################################################
    #Predict on final test data
    X_test_final = X_test_final.astype(np.float32)
    x_test_final = torch.from_numpy(X_test_final)
    y_pred_final = model(x_test_final)
    pyT_predictions_final = y_pred_final.detach().numpy()
    #print(pyT_predictions_final)
    #pyT_predictions_final = (np.argmax(pyT_predictions_final,1) + 1)


    ###################################################################
    #write results to file

    names = X_full[:,0].reshape(-1,1)
    names = names.astype(int)
    names = names.astype(str)
    names = np.core.defchararray.add('ISIC00', names)
    labels_final = pyT_predictions_final.reshape(-1,7)
    results_to_write = np.concatenate((names, labels_final), axis = 1)
    print('final results size: {}'.format(results_to_write.shape))
    df = pd.DataFrame(results_to_write)
    df.to_csv("file_path.csv", header = None, index=None)
    
    
    ###################################################################
    # # Create a linear SVM classifier with C = 1
    # clf = svm.SVC(kernel='linear', C=1)
    # # Create SVM classifier based on RBF kernel. 
    # clf = svm.SVC(kernel='rbf', C = 10.0, gamma=0.1)
    # # Grid Search
    # # Parameter Grid
    # param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    
    # # Make grid search classifier
    # clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
    
    # # Train the classifier
    # clf_grid.fit(X_train, y_train)
    
    # # clf = grid.best_estimator_()
    # print("Best Parameters:\n", clf_grid.best_params_)
    # print("Best Estimators:\n", clf_grid.best_estimator_)


test()

