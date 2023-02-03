import numpy as np
import cv2 as cv
import csv

from matplotlib import pyplot as plt

#When using get_pca1
from matplotlib.mlab import PCA

#When using get_pca2
from numpy import mean,cov,cumsum,dot,linalg,size,flipud

#When using get_pca3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Assignments
X=[]
img_number=24306
histr = []


def get_histogram():

    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [25], [0, 256])
        #plt.plot(histr, color=col)
        #plt.xlim([0, 256])
        #print(histr)
        # get_pca1()
        global X
        X=np.append(X, histr)
    #plt.show()


def get_pca1():  #USES SVD(Sigular Value Decomposition); Memory and Processor intensive
    data = np.array(histr)
    results = PCA(data)
    print (results.Y)


def get_pca2(A,numpc=0):
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))
    p = size(coeff,axis=1)
    idx = argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    if numpc < p and numpc >= 0:
        coeff = coeff[:,range(numpc)] # cutting some PCs if needed
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent


def get_pca3(A):
    standardizedData = StandardScaler().fit_transform(histr)

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X = standardizedData)

    # to get how much variance was retained
    print(pca.explained_variance_ratio_.sum())



for j in range(10015): #10015
    img = cv.imread('../data/TRAINING_DATA/ISIC2018_Task3_Training_Input/ISIC_00'+str(img_number)+'.jpg', 1)
    color = ('b', 'g', 'r')
    get_histogram()
    img_number=img_number+1
#get_pca1()

X=X.reshape(-1, 25*3)

myFile = open('../data/TRAINING_DATA/histo_features_25.csv', 'w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(X)
     
print("Writing complete")

print(X.shape)
print(type(X))
