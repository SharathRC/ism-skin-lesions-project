import numpy as np
import csv
import pandas as pd 

import math, random

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
import torch.utils.data
import torch.cuda._utils 
#import torch.cuda.utils.data

from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

torch.cuda.set_device(0)

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
    print('Y test shape: {}'.format(Y_test.shape))
    print(pyT_predictions.shape)
    confusion_mat = confusion_matrix(Y_test, pyT_predictions)
    print(confusion_mat)
    print(len(confusion_matrix))

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
    
    #split_factor = [8, 35, 2.8, 2, 5, 1.1, 1.1]
    split_factor = [2, 6, 1.5, 1.5, 1.5, 1.1, 1.1]
    
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
    
    # R Neural network PYTORCH
      
    class RNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
            super(RNNModel, self).__init__()
            # Hidden dimensions
            self.hidden_dim = hidden_dim

            # Number of hidden layers
            self.layer_dim = layer_dim

            # Building your RNN
            # batch_first=True causes input/output tensors to be of shape
            # (batch_dim, seq_dim, input_dim)
            # batch_dim = number of samples per batch
            self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

            # Readout layer
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            # (layer_dim, batch_size, hidden_dim)
            #h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
            # We need to detach the hidden state to prevent exploding/vanishing gradients
            # This is part of truncated backpropagation through time (BPTT)
            
            #out, hn = self.rnn(x, h0.detach())
            
            out, hn = self.rnn(x, h0)
            # Index hidden state of last time step
            # out.size() --> 100, 28, 10
            # out[:, -1, :] --> 100, 10 --> just want last time step hidden states! 
            out = self.fc(out[:, -1, :]) 
            # out.size() --> 100, 10
            return out

    n_in, n_h1, n_h2, n_h3, n_out, batch_size, n_epochs, n_iters = num_features, 75, 45, 15, nlabels, num_samples, 100, 50

    X_train = X_train.astype(np.float32)
    Y_train_pyT = Y_train_pyT.astype(np.float32)
    
    X_test = X_test.astype(np.float32)
    Y_test_pyT = Y_test_pyT.astype(np.float32)

    X_test_final = X_test_final.astype(np.float32)
    
    #x = torch.cuda.randn(batch_size, n_in)
    #y = torch.cuda.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0]])
    #X_train_RNN = np.ravel(X_train)
    #X_train_RNN = X_train_RNN.reshape(-1, 1, num_features)
    #print('X_TRAIN_RNN shape: {}'.format(X_train_RNN.shape))
    #X_test_RNN = np.ravel(X_test)
    #X_test_RNN = X_test_RNN.reshape(-1, 1, num_features)
    
    batch_size = 10
    n_iters = 250000
    num_epochs = n_iters / (len(X_train) / batch_size)
    num_epochs = int(num_epochs) * 2

    xTrain = torch.from_numpy(X_train)
    yTrain = torch.from_numpy(Y_train_pyT)#.type(torch.cuda.LongTensor)
    print(xTrain.size())

    xTest = torch.from_numpy(X_test)
    yTest = torch.from_numpy(Y_test_pyT).type(torch.cuda.LongTensor)

    xTest_FINAL = torch.from_numpy(X_train)
    # bochs = int(num_epochs)

    train = torch.utils.data.TensorDataset(xTrain, yTrain)
    test = torch.utils.data.TensorDataset(xTest, yTest)
    test_FINAL = torch.utils.data.TensorDataset(xTest_FINAL)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = 1, shuffle = False) 
    test_FINAL_loader = torch.utils.data.DataLoader(test_FINAL, batch_size = 1, shuffle = False)  
    

    input_dim = 94    # input dimension
    hidden_dim = 75  # hidden layer dimension
    layer_dim = 1     # number of hidden layers
    output_dim = 7   # output dimension

    #model = RNNModel(n_in, n_h1, 1, n_out)

    model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

    # Cross Entropy Loss 
    error = nn.MSELoss()  #CrossEntropyLoss, MSELoss, MultiLabelSoftMarginLoss

    # SGD Optimizer
    learning_rate = 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.MultiLabelSoftMarginLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #SGD
    print(model)
    
    seq_dim = 1  
    loss_list = []
    iteration_list = []
    accuracy_list = []
    count = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            train  = Variable(images.view(-1, seq_dim, input_dim))
            #print(train)
            labels = Variable(labels )
            #print(labels)
                
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            count += 1
            
            # if count % 250 == 0:
            #     # Calculate Accuracy         
            #     correct = 0
            #     total = 0
            #     # Iterate through test dataset
            #     for images, labels in test_loader:
            #         images = Variable(images.view(-1, seq_dim, input_dim))
                    
            #         # Forward propagation
            #         outputs = model(images)
                    
            #         # Get predictions from the maximum value
            #         predicted = torch.cuda.max(outputs.data, 1)[1]
                    
            #         # Total number of labels
            #         total += labels.size(0)
                    
            #         correct += (predicted == labels).sum()
                
            #     accuracy = 100 * correct / float(total)
                
            #     # store loss and iteration
            #     loss_list.append(loss.data)
            #     iteration_list.append(count)
            #     accuracy_list.append(accuracy)
            #     if count % 500 == 0:
            #         # Print Loss
            #         print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data[0], accuracy))
    print('---------------------------------------------------------')
    print('Training complete')
    print('---------------------------------------------------------')

    pyT_predictions_RNN = []
    value = 0
    correct = 0
    total = 0
    count = 0
    for images, labels in test_loader:
        count+= 1
        images = Variable(images.view(-1, seq_dim, input_dim))
        #print(images)
        #print(images.shape)
        
        #print(labels)
        #print(labels.shape)
        # Forward propagation
        outputs = model(images)
        
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        print(predicted)
        value = predicted.numpy()[0]
        #print(predicted.)
        pyT_predictions_RNN = np.append(pyT_predictions_RNN, value)
        # Total number of labels
        total += labels.size(0)
        
        #correct += (predicted == labels).sum()
    
    print(count)
    print(pyT_predictions_RNN.shape)
    #accuracy = 100 * correct / float(total)
    
    # store loss and iteration
    #loss_list.append(loss.data)
    #iteration_list.append(count)
    #accuracy_list.append(accuracy)
    
    #print('Accuracy: {}'.format(accuracy))
    
    #y_pred = model(xTest)
    #pyT_predictions = y_pred.detach().numpy()
    #pyT_predictions = (np.argmax(pyT_predictions,1) + 1)
    
    #pyT_predictions_RNN = pyT_predictions_RNN.detach().numpy()
    print(Y_test.shape)
    print('predictions: {}'.format(pyT_predictions_RNN))
    print('groung truth: {}'.format(Y_test))

    #pyT_predictions = pyT_predictions >= 0.5
    score = fbeta_score(Y_test, pyT_predictions_RNN, average='weighted', beta=0.5) #None, binary (default), micro, macro, samples, weighted
    #print("Accuracy (from pytorch.cuda): {}%".format( score * 100 ))
    print(score)
    calc_accuracy(Y_test, pyT_predictions_RNN)
    
    ###################################################################
    #Predict on final test data
    pyT_predictions_FINAL_RNN = []
    
    correct = 0
    total = 0
    count = 0
    for images in test_FINAL_loader:
        count+= 1
        images = Variable(images.view(-1, seq_dim, input_dim))
        #print(images)
        #print(images.shape)
        
        #print(labels)
        #print(labels.shape)
        # Forward propagation
        outputs = model(images)
        
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]
        #print(predicted.)
        pyT_predictions_FINAL_RNN.append(predicted)
        # Total number of labels
        total += labels.size(0)
        
        correct += (predicted == labels).sum()
    
    print(count)
    accuracy = 100 * correct / float(total)

    #pyT_predictions_final = pyT_predictions_FINAL_RNN.detach().numpy()
    pyT_predictions_final_rnn = (np.argmax(pyT_predictions_FINAL_RNN,1) + 1)


    ###################################################################
    #write results to file

    names = X_full[:,0].reshape(-1,1)
    names = names.astype(int)
    names = names.astype(str)
    names = np.core.defchararray.add('ISIC00', names)
    labels_final = pyT_predictions_final_rnn.reshape(-1,1)
    labels_final = labels_final.astype(int)
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