import numpy as np
import scipy.optimize as op
import csv
from f1score_calc import f1score_calc
from predict import predict

NUM_OF_FEATURES = 3

def write_labels():
    path = '../data/'
    myfile = open( path +"ISIC2018_Task3_Training_GroundTruth.csv", "r")
    reader = csv.reader(myfile)
    
    t = []
    y = []
    data = []
    c = 0
    for line in reader:
        c+= 1
        if c == 1:
            continue
        y = 0
        if float(line[1]) == 1 or float(line[3]) == 1 or float(line[4]) == 1:
            y = 1
        
        line.append(y)
        data.append(line)

    print(data)
    myfile2 = open('test_data.csv', 'w')
    with myfile:
        writer = csv.writer(myfile2)
        writer.writerows(data)  
    
    print("Writing complete") 

def write_columns():
    path = '../data/TRAINING_DATA/train_data.csv'
    myfile = open( path, "r")
    reader = csv.reader(myfile)
    
    t = []
    y = []
    data = []
    c = 0
    for line in reader:
        data = np.append(data, line)
    
    print(data.shape)

    path = '../data/TRAINING_DATA/histo_features.csv'
    myfile = open( path, "r")
    reader = csv.reader(myfile)
    
   
    X = []
    c = 0
    for line in reader:
        data = np.append(data, line)

    data = np.concatenate( ( data, X), axis = 1 )
    print(data.shape)
    myfile2 = open('train_data.csv', 'w')
    with myfile:
        writer = csv.writer(myfile2)
        writer.writerows(data)  
    
    print("Writing complete")

write_columns()
