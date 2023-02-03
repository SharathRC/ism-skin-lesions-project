import numpy as np
import cv2, sys, csv, os
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from centroid import centroid
from skimage.feature import greycomatrix, greycoprops

MAX_KERNEL_LENGTH = 31
src = None
imgray = None
compressed_img_kmeans = None
blur_img = None
one_contour = []
X=[]


##########################################################################################################################
file_path_img = '../data/TRAINING_DATA/ISIC2018_Task3_Training_Input_hair_removed/ISIC_00'  #train
#file_path_img = '../data/TEST_DATA/ISIC2018_Task3_Test_Input_hair_removed/ISIC_00' #test

file_path_csv = '../data/TRAINING_DATA/features_25hist_contour_otsu_12texture_around_contour_hair_removed_ordered.csv' #train
#file_path_csv = '../data/TEST_DATA/features_25hist_contour_otsu_12texture_around_contour_hair_removed_test_ordered.csv' #test

NUM_IMG = 10016 #train
#NUM_IMG = 1540 #test

step = 1
img_number = 24306 - step # train   
#img_number = 34524 - step # test

##########################################################################################################################
PIXEL = 5000
histr = []
img_names = []
NUM_FEATURES = 95


def load_image():
    global src, imgray, height, width, img_center, xmax, xmin, ymax, ymin
    src = cv2.imread('../data/TRAINING_DATA/ISIC2018_Task3_Training_Input/ISIC_0024322.jpg')
    imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width, l = src.shape
    img_center = np.array([width/2, height/2])
    xmax = width*3/4
    xmin = width*1/4
    ymax = height*3/4
    ymin = height*1/4
    print('width: {}, xmin: {}, xmax: {}, height: {}, ymin: {}, ymax: {}'.format(width, xmin, xmax, height, ymin, ymax))
    
def compress(src, K):
    global compressed_img_kmeans
    Z = src.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)  
    center = np.array(center)   
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    compressed_img_kmeans = res.reshape(src.shape)
    return compressed_img_kmeans

def get_contour(z):
    global thresh, imgray, img_equ
    imgray = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)     #COLOR_BGR2GRAY     COLOR_BGR2YCrCb        COLOR_BGR2HSV
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_equ = clahe.apply(imgray)
    #Equalize image
    #imgray=cv2.equalizeHist(imgray)
    #ret, thresh = cv2.threshold(img_equ, 155, 255, cv2.THRESH_BINARY_INV)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #thresh=cv2.adaptiveThreshold(img_equ, 255, cv2.THRESH_BINARY_INV, cv2.ADAPTIVE_THRESH_MEAN_C, 351, 0)
    #thresh=cv2.adaptiveThreshold(imgray, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY_INV, 11, 0)

    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp_img = np.copy(z)
    #cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)
    #cv2.imshow('Contours', temp_img)

    global contours_modified
    contours_modified=[]
    lst_dist_to_center = []
    dist_to_center = 0
    area = 0
    one_contour = []
    #flag = True
    for i in range(len(contours)):
        m, l, n= contours[i].shape
        if m>200:
            cont_center = centroid(contours[i].reshape(-1,2))
            dist_to_center = np.linalg.norm( cont_center - img_center )
            area = cv2.contourArea(contours[i])
            #print('{}: {}'.format('in if 1', cont_center))
            if cont_center[0] >= xmin and cont_center[0] <= xmax and cont_center[1] >= ymin and cont_center[1] <= ymax :
                lst_dist_to_center.append(dist_to_center)
                contours_modified.append(contours[i])
                #print('in if 2')
    
    #print('{}: {}'.format('no. of contours', len(contours)))
    #print('{}: {}'.format('no. of contours_modified', len(contours_modified)))
    blank_image = np.zeros((height,width,3), np.uint8)
    #cv2.drawContours(blank_image, contours_modified, -1, (255, 0, 0), 3)
    #cv2.imshow('Contours Modified', blank_image)
    try:
        m = np.argmin(lst_dist_to_center)
    except:
        return one_contour
    one_contour = contours_modified[m]
    area = cv2.contourArea(one_contour)

    blank_image = np.zeros((height,width,3), np.uint8)
    #cv2.drawContours(blank_image, one_contour, -1, (255, 0, 0), 0)
    #cv2.imshow('ONE CONTOUR', blank_image)
    
    #print('{}: {}'.format('length of final contour', len(one_contour)))
    #print('{}: {}'.format('area of final contour', area))

    return one_contour

def get_moments(contour):
    retval = cv2.moments(contour)
    hu = cv2.HuMoments(retval)
    hu = hu.reshape(1,-1)
    global X
    X=np.append(X, hu)
    return hu

def get_mask(contour_mask, my_image):
    height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)
    result_inv = np.full(my_image.shape, 255, my_image.dtype)

    cv2.fillPoly(result,pts=[contour_mask],color=[255,255,255])
    cv2.fillPoly(result_inv,pts=[contour_mask],color=[0,0,0])
    '''
    m, n = contour_mask.shape
    for i in range(m):
        result[contour_mask[i][1],contour_mask[i][0]]=[255,255,255]
    '''
    #cv2.imshow('CONTOUR MASK', result)

    return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.cvtColor(result_inv, cv2.COLOR_BGR2GRAY)

def get_histo(mask, image):
    # create a mask
    # mask = np.zeros(img.shape[:2], np.uint8)
    # mask[100:300, 100:400] = 255
    # mask[points[0]] = 255
    global X, img_names, img_number, masked_img, hist_full, hist_mask
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    hist_mask_lst = []
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist_full = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_mask = cv2.calcHist([image], [i], mask, [25], [0, 256])
        hist_mask_lst=np.append(hist_mask_lst, hist_mask)
    #     plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     plt.subplot(222), plt.imshow(mask, 'gray')
    #     plt.subplot(223), plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    #     plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    #     plt.xlim([0, 256])
    # plt.show()
    num_pixel = np.sum(hist_mask_lst)
    hist_mask_lst = hist_mask_lst * (PIXEL/num_pixel)
    X=np.append(X, img_number )
    X=np.append(X, hist_mask_lst)

def get_texture(image):
    pixel_distance = 10
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_compressed = compress(gray, 64)
    glcm = greycomatrix(gray, [pixel_distance], angles, 256, symmetric=True, normed=True)
    #print(gray.shape, type(gray))
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    correlation = greycoprops(glcm, 'correlation')
    ASM = greycoprops(glcm, 'ASM')
    energy = greycoprops(glcm, 'energy')
    contrast = greycoprops(glcm, 'contrast')

    diss_avg = (dissimilarity[0][0]+dissimilarity[0][1]+dissimilarity[0][2]+dissimilarity[0][3])/4
    homg_avg = (homogeneity[0][0]+homogeneity[0][1]+homogeneity[0][2]+homogeneity[0][3])/4
    corr_avg = (correlation[0][0]+correlation[0][1]+correlation[0][2]+correlation[0][3])/4
    ASM_avg = (ASM[0][0]+ASM[0][1]+ASM[0][2]+ASM[0][3])/4
    energy_avg = (energy[0][0]+energy[0][1]+energy[0][2]+energy[0][3])/4
    contrast_avg = (contrast[0][0]+contrast[0][1]+contrast[0][2]+contrast[0][3])/4
    
    dissimilarity = dissimilarity.reshape(1,-1)
    homogeneity = homogeneity.reshape(1,-1)
    correlation = correlation.reshape(1,-1)
    ASM = ASM.reshape(1,-1)
    energy = energy.reshape(1,-1)
    contrast = contrast.reshape(1,-1)
    
    diss_range = np.ptp(dissimilarity,axis=1)
    homg_range = np.ptp(homogeneity,axis=1)
    corr_range = np.ptp(correlation,axis=1)
    ASM_range = np.ptp(ASM,axis=1)
    energy_range = np.ptp(energy,axis=1)
    contrast_range = np.ptp(contrast,axis=1)
    
    global X
    X=np.append(X, diss_avg)
    X=np.append(X, homg_avg)
    X=np.append(X, corr_avg)
    X=np.append(X, ASM_avg)
    X=np.append(X, energy_avg)
    X=np.append(X, contrast_avg)
    
    X=np.append(X, diss_range)
    X=np.append(X, homg_range)
    X=np.append(X, corr_range)
    X=np.append(X, ASM_range)
    X=np.append(X, energy_range)
    X=np.append(X, contrast_range)

    

    # print(dissimilarity, homogeneity, correlation, ASM, energy, contrast)
    # print(diss_avg, homg_avg, corr_avg, ASM_avg, energy_avg, contrast_avg)
    # print(diss_range, homg_range, corr_range, ASM_range, energy_range, contrast_range)
    # print(diss_range)
    #cv2.imshow('Textures Image', gray)

def get_texture_contour(image, contour):
    pixel_distance = 1
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    max_level = 256
    matrix_size = 9
    offset = int((matrix_size - 1) / 2)

    dissimilarity = np.array([0, 0, 0, 0]).reshape(1, -1)
    homogeneity = np.array([0, 0, 0, 0]).reshape(1, -1)
    correlation = np.array([0, 0, 0, 0]).reshape(1, -1)
    ASM = np.array([0, 0, 0, 0]).reshape(1, -1)
    energy = np.array([0, 0, 0, 0]).reshape(1, -1)
    contrast = np.array([0, 0, 0, 0]).reshape(1, -1)

    tot_sub_diss = 0
    tot_sub_homg = 0
    tot_sub_corr = 0
    tot_sub_ASM = 0
    tot_sub_energy = 0
    tot_sub_contrast = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sub_img = np.zeros((matrix_size, matrix_size), gray.dtype)

    m, n = contour.shape
    for i in range(m):
        center = contour[i]
        for j in range(matrix_size):
            for k in range(matrix_size):
                sub_img[j][k] = gray[center[1] - offset + j][center[0] - offset + k]

        sub_img = sub_img.reshape(matrix_size, matrix_size)
        sub_glcm = greycomatrix(sub_img, [pixel_distance], angles, max_level, symmetric=True, normed=True)

        sub_dissimilarity = greycoprops(sub_glcm, 'dissimilarity')
        sub_homogeneity = greycoprops(sub_glcm, 'homogeneity')
        sub_correlation = greycoprops(sub_glcm, 'correlation')
        sub_ASM = greycoprops(sub_glcm, 'ASM')
        sub_energy = greycoprops(sub_glcm, 'energy')
        sub_contrast = greycoprops(sub_glcm, 'contrast')

        sub_diss_avg = (sub_dissimilarity[0][0] + sub_dissimilarity[0][1] + sub_dissimilarity[0][2] +
                        sub_dissimilarity[0][3]) / 4
        sub_homg_avg = (sub_homogeneity[0][0] + sub_homogeneity[0][1] + sub_homogeneity[0][2] + sub_homogeneity[0][
            3]) / 4
        sub_corr_avg = (sub_correlation[0][0] + sub_correlation[0][1] + sub_correlation[0][2] + sub_correlation[0][
            3]) / 4
        sub_ASM_avg = (sub_ASM[0][0] + sub_ASM[0][1] + sub_ASM[0][2] + sub_ASM[0][3]) / 4
        sub_energy_avg = (sub_energy[0][0] + sub_energy[0][1] + sub_energy[0][2] + sub_energy[0][3]) / 4
        sub_contrast_avg = (sub_contrast[0][0] + sub_contrast[0][1] + sub_contrast[0][2] + sub_contrast[0][3]) / 4

        tot_dissimilarity = dissimilarity + sub_dissimilarity
        tot_homogeneity = homogeneity + sub_homogeneity
        tot_correlation = correlation + sub_correlation
        tot_ASM = ASM + sub_ASM
        tot_energy = energy + sub_energy
        tot_contrast = contrast + sub_contrast

        tot_sub_diss = sub_diss_avg + tot_sub_diss
        tot_sub_homg = sub_homg_avg + tot_sub_homg
        tot_sub_corr = sub_corr_avg + tot_sub_corr
        tot_sub_ASM = sub_ASM_avg + tot_sub_ASM
        tot_sub_energy = sub_energy_avg + tot_sub_energy
        tot_sub_contrast = sub_contrast_avg + tot_sub_contrast

    dissimilarity = tot_dissimilarity / m
    homogeneity = tot_homogeneity / m
    correlation = tot_correlation / m
    ASM = tot_ASM / m
    energy = tot_energy / m
    contrast = tot_contrast / m

    diss_avg = tot_sub_diss / m
    homg_avg = tot_sub_homg / m
    corr_avg = tot_sub_corr / m
    ASM_avg = tot_sub_ASM / m
    energy_avg = tot_sub_energy / m
    contrast_avg = tot_sub_contrast / m

    dissimilarity = dissimilarity.reshape(1, -1)
    homogeneity = homogeneity.reshape(1, -1)
    correlation = correlation.reshape(1, -1)
    ASM = ASM.reshape(1, -1)
    energy = energy.reshape(1, -1)
    contrast = contrast.reshape(1, -1)

    diss_range = np.ptp(dissimilarity, axis=1)
    homg_range = np.ptp(homogeneity, axis=1)
    corr_range = np.ptp(correlation, axis=1)
    ASM_range = np.ptp(ASM, axis=1)
    energy_range = np.ptp(energy, axis=1)
    contrast_range = np.ptp(contrast, axis=1)

    global X
    X = np.append(X, diss_avg)
    X = np.append(X, homg_avg)
    X = np.append(X, corr_avg)
    X = np.append(X, ASM_avg)
    X = np.append(X, energy_avg)
    X = np.append(X, contrast_avg)

    X = np.append(X, diss_range)
    X = np.append(X, homg_range)
    X = np.append(X, corr_range)
    X = np.append(X, ASM_range)
    X = np.append(X, energy_range)
    X = np.append(X, contrast_range)

    #print(dissimilarity, homogeneity, correlation, ASM, energy, contrast)
    #print(diss_avg, homg_avg, corr_avg, ASM_avg, energy_avg, contrast_avg)
    #print(diss_range, homg_range, corr_range, ASM_range, energy_range, contrast_range)
    # cv2.imshow('Textures Image', gray)

def get_texture_inner_contour(image, contour):
    pixel_distance = 1
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    max_level = 256
    matrix_size = 15
    offset = (matrix_size - 1) / 2
    cententroid = centroid(contour) 

    dissimilarity = np.array([0, 0, 0, 0]).reshape(1, -1)
    homogeneity = np.array([0, 0, 0, 0]).reshape(1, -1)
    correlation = np.array([0, 0, 0, 0]).reshape(1, -1)
    ASM = np.array([0, 0, 0, 0]).reshape(1, -1)
    energy = np.array([0, 0, 0, 0]).reshape(1, -1)
    contrast = np.array([0, 0, 0, 0]).reshape(1, -1)

    tot_sub_diss = 0
    tot_sub_homg = 0
    tot_sub_corr = 0
    tot_sub_ASM = 0
    tot_sub_energy = 0
    tot_sub_contrast = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sub_img = np.zeros((matrix_size, matrix_size), gray.dtype)

    m, n = contour.shape
    for i in range(m):
        center = contour[i]
        for j in range(15):
            for k in range(15):
                sub_img[j][k] = gray[center[1] - offset + j][center[0] - offset + k]

        sub_img = sub_img.reshape(15, 15)
        sub_glcm = greycomatrix(sub_img, [pixel_distance], angles, max_level, symmetric=True, normed=True)

        sub_dissimilarity = greycoprops(sub_glcm, 'dissimilarity')
        sub_homogeneity = greycoprops(sub_glcm, 'homogeneity')
        sub_correlation = greycoprops(sub_glcm, 'correlation')
        sub_ASM = greycoprops(sub_glcm, 'ASM')
        sub_energy = greycoprops(sub_glcm, 'energy')
        sub_contrast = greycoprops(sub_glcm, 'contrast')

        sub_diss_avg = (sub_dissimilarity[0][0] + sub_dissimilarity[0][1] + sub_dissimilarity[0][2] + sub_dissimilarity[0][3]) / 4
        sub_homg_avg = (sub_homogeneity[0][0] + sub_homogeneity[0][1] + sub_homogeneity[0][2] + sub_homogeneity[0][3]) / 4
        sub_corr_avg = (sub_correlation[0][0] + sub_correlation[0][1] + sub_correlation[0][2] + sub_correlation[0][3]) / 4
        sub_ASM_avg = (sub_ASM[0][0] + sub_ASM[0][1] + sub_ASM[0][2] + sub_ASM[0][3]) / 4
        sub_energy_avg = (sub_energy[0][0] + sub_energy[0][1] + sub_energy[0][2] + sub_energy[0][3]) / 4
        sub_contrast_avg = (sub_contrast[0][0] + sub_contrast[0][1] + sub_contrast[0][2] + sub_contrast[0][3]) / 4

        tot_dissimilarity = dissimilarity + sub_dissimilarity
        tot_homogeneity = homogeneity + sub_homogeneity
        tot_correlation = correlation + sub_correlation
        tot_ASM = ASM + sub_ASM
        tot_energy = energy + sub_energy
        tot_contrast = contrast + sub_contrast

        tot_sub_diss = sub_diss_avg + tot_sub_diss
        tot_sub_homg = sub_homg_avg + tot_sub_homg
        tot_sub_corr = sub_corr_avg + tot_sub_corr
        tot_sub_ASM = sub_ASM_avg + tot_sub_ASM
        tot_sub_energy = sub_energy_avg + tot_sub_energy
        tot_sub_contrast = sub_contrast_avg + tot_sub_contrast

    dissimilarity = tot_dissimilarity / m
    homogeneity = tot_homogeneity / m
    correlation = tot_correlation / m
    ASM = tot_ASM / m
    energy = tot_energy / m
    contrast = tot_contrast / m

    diss_avg = tot_sub_diss / m
    homg_avg = tot_sub_homg / m
    corr_avg = tot_sub_corr / m
    ASM_avg = tot_sub_ASM / m
    energy_avg = tot_sub_energy / m
    contrast_avg = tot_sub_contrast / m

    dissimilarity = dissimilarity.reshape(1, -1)
    homogeneity = homogeneity.reshape(1, -1)
    correlation = correlation.reshape(1, -1)
    ASM = ASM.reshape(1, -1)
    energy = energy.reshape(1, -1)
    contrast = contrast.reshape(1, -1)

    diss_range = np.ptp(dissimilarity, axis=1)
    homg_range = np.ptp(homogeneity, axis=1)
    corr_range = np.ptp(correlation, axis=1)
    ASM_range = np.ptp(ASM, axis=1)
    energy_range = np.ptp(energy, axis=1)
    contrast_range = np.ptp(contrast, axis=1)

    global X
    X = np.append(X, diss_avg)
    X = np.append(X, homg_avg)
    X = np.append(X, corr_avg)
    X = np.append(X, ASM_avg)
    X = np.append(X, energy_avg)
    X = np.append(X, contrast_avg)

    X = np.append(X, diss_range)
    X = np.append(X, homg_range)
    X = np.append(X, corr_range)
    X = np.append(X, ASM_range)
    X = np.append(X, energy_range)
    X = np.append(X, contrast_range)

    print(dissimilarity, homogeneity, correlation, ASM, energy, contrast)
    print(diss_avg, homg_avg, corr_avg, ASM_avg, energy_avg, contrast_avg)
    print(diss_range, homg_range, corr_range, ASM_range, energy_range, contrast_range)
    # cv2.imshow('Textures Image', gray)

def write_features():
    global img_number, X, src, step, mole_mask_inv, mole_isolated
    
    for j in range(1,NUM_IMG,step): #10015
        img_number=img_number+step       
        src = cv2.imread(file_path_img + str(img_number)+'.jpg', 1)  
        if src is None:
            continue
        print(img_number)  
        global height, width, img_center, xmax, xmin, ymax, ymin
        height, width, l = src.shape
        img_center = np.array([height/2, width/2])
        xmax = width*3/4
        xmin = width*1/4
        ymax = height*3/4
        ymin = height*1/4      
        i = 30
        blur_img = cv2.bilateralFilter(src, i, i * 2, i / 2)
        contour_mole = get_contour(blur_img)
        
        if len(contour_mole) == 0:
            continue       
        
        contour_mole = contour_mole.reshape(-1, 2)
        mole_mask, mole_mask_inv = get_mask(contour_mole, src)
        mole_isolated = cv2.bitwise_and(src, src, mask=mole_mask)
        get_histo(mole_mask, src)
        get_moments(contour_mole)
        #get_texture(mole_isolated)     
        get_texture_contour(src, contour_mole)    
    
    X = X.reshape(-1, NUM_FEATURES)
    myFile = open(file_path_csv, 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(X)       
    print("Writing complete")
    print(X.shape)
    print(type(X))

def test_image():
    global src, blur_img, mole_mask, mole_mask_inv, mole_isolated, mole_isolated, imgray
    load_image()   
    #compressed_img_kmeans = compress(src, 16)
    i = 31
    blur_img = cv2.bilateralFilter(src, i, i * 2, i / 2)    
    
    contour_mole = get_contour(blur_img)
    get_moments(contour_mole)
    
    contour_mole = contour_mole.reshape(-1, 2)
    mole_mask, mole_mask_inv = get_mask(contour_mole, src)
    mole_isolated = cv2.bitwise_and(src, src, mask=mole_mask)
    get_histo(mole_mask, src)
    hist_mask = cv2.calcHist([imgray],[0],mole_mask,[25],[0,256])
    print(hist_mask.shape)
    get_texture(mole_isolated)

def test_images_from_folder(folder):
    global img_number, X, src, step, mole_mask_inv, mole_isolated
    for filename in os.listdir(folder):
        src = cv2.imread(os.path.join(folder,filename))
        if src is not None:
            img_name = filename.split( "00", 1 )           
            img_number = int(img_name[1].split( ".", 1 )[0])                  
            print(img_number)  
            global height, width, img_center, xmax, xmin, ymax, ymin
            height, width, l = src.shape
            img_center = np.array([height/2, width/2])
            xmax = width*3/4
            xmin = width*1/4
            ymax = height*3/4
            ymin = height*1/4      
            i = 30
            blur_img = cv2.bilateralFilter(src, i, i * 2, i / 2)
            contour_mole = get_contour(blur_img)
            
            if len(contour_mole) == 0:
                continue       
            
            contour_mole = contour_mole.reshape(-1, 2)
            mole_mask, mole_mask_inv = get_mask(contour_mole, src)
            mole_isolated = cv2.bitwise_and(src, src, mask=mole_mask)
            get_histo(mole_mask, src)
            get_moments(contour_mole)
            get_texture(mole_isolated)
    
    X = X.reshape(-1, NUM_FEATURES)
    myFile = open(file_path_csv, 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(X)       
    print("Writing complete")
    print(X.shape)
    print(type(X))

def show_img():
    global src, blur_img
    cv2.imshow('Original',src)
    cv2.imshow('bilateral_filter', blur_img)
    cv2.imshow('gray', imgray)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('CLAHE', img_equ)
    cv2.imshow('Mask', mole_mask)
    cv2.imshow('Mask Inverted', mole_mask_inv)
    cv2.imshow('mole_isolated', mole_isolated)

    if compressed_img_kmeans is not None:
        cv2.imshow('Compressed_kmeans', compressed_img_kmeans)

    #     plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     plt.subplot(222), plt.imshow(mask, 'gray')
    #     plt.subplot(223), plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    #     plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    #     plt.xlim([0, 256])
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#folder = '../data/TRAINING_DATA/ISIC2018_Task3_Training_Input_hair_removed/'
folder = '../data/TEST_DATA/ISIC2018_Task3_Test_Input_hair_removed/'
#test_images_from_folder(folder)
write_features()
#test_image()
#show_img()

