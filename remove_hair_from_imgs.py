import numpy as np
import cv2, sys, csv, os
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from dullRazor import remove_hair

MAX_KERNEL_LENGTH = 31
src = None
imgray = None
compressed_img_kmeans = None
blur_img = None
one_contour = []
X=[]

file_path_img = '../data/TRAINING_DATA/ISIC2018_Task3_Training_Input/ISIC_00'  #train
#file_path_img = '../data/TEST_DATA/ISIC2018_Task3_Test_Input/ISIC_00' #test

out_file_path = '../data/TRAINING_DATA/ISIC2018_Task3_Training_Input_hair_removed/ISIC_00' #train
#out_file_path = '../data/TEST_DATA/ISIC2018_Task3_Test_Input_hair_removed/ISIC_00' #test

NUM_IMG = 10016 #train
#NUM_IMG = 1541 #test

step = 1

img_number = 34320 - step # train   24306
#img_number = 34524 - step # test 34524


def write_features():
    global img_number, X, src, step, mole_mask_inv, mole_isolated
    count = 0
    for j in range(1,NUM_IMG,step): #10015
        img_number = img_number+step  
        count+=1 
        
        #if count == 1000:
        #    break    
        if not 0 <= count < 4:
            continue
        
        src = cv2.imread(file_path_img + str(img_number)+'.jpg', 1)  
        if src is None:
            continue
        print(img_number)

        modified_src = remove_hair(src)
        cv2.imwrite(out_file_path + str(img_number)+'.jpg', modified_src)
        

write_features()
