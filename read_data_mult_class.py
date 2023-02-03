import numpy as np
import csv

NUM_OF_FEATURES = 0

def read_data_multi(step, pathX, flag):
    global NUM_OF_FEATURES
    #path = '../data/TRAINING_DATA/histo_features_25.csv'
    #path = '../data/TRAINING_DATA/features_25hist_contour_otsu_12texture.csv'
    myfile = open( pathX, "r")
    reader = csv.reader(myfile)
    t = []
    row = []
    count = 0
    img_names = []
    for line in reader:
        count +=1
        if not count % step == 0 :
            continue       
        row = [float(i) for i in line[1:]]
        NUM_OF_FEATURES = len(row)
        t = np.append(t, row)
        img_names = np.append(img_names, float(line[0]))   
    X = t.reshape(-1,NUM_OF_FEATURES)
    
    if flag:
        path = '../data/TRAINING_DATA/train_data.csv'
        myfile = open( path, "r")
        reader = csv.reader(myfile)
        y = []
        count = 24305
        for line in reader:
            count +=1
            if count not in img_names:
                continue
            #label = float(line[8])
            #label = [float(i) for i in line[1:8]]
            label = (np.argmax(line[1:8]) + 1)
            y = np.append(y, label)
        

        #t = [float(i) for i in t]
        
        Y = y.reshape(-1,1)
        return X, Y
    
    else:
        return X
