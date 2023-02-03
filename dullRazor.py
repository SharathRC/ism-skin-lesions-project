import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
#### Here we should load all images one by one
def remove_hair(image):
        #file_path_img = '../data/TRAINING_DATA/ISIC2018_Task3_Training_Input/ISIC_0024356.jpg'  #train
        #image = cv.imread(file_path_img)
        #plt.subplot(121), plt.imshow(image)

        #### Rest will do the job
        final_img = image.copy()
        b = image.copy()
        # set green and red channels to 0. Convert Gray image
        b[:, :, 1] = 0
        b[:, :, 2] = 0
        b_gray = cv.cvtColor(b,cv.COLOR_BGR2GRAY)
        g = image.copy()
        # set blue and red channels to 0. Convert Gray image
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        g_gray = cv.cvtColor(g,cv.COLOR_BGR2GRAY)
        r = image.copy()
        # set blue and green channels to 0. Convert Gray image
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        r_gray = cv.cvtColor(r,cv.COLOR_BGR2GRAY)
        # Kernel Structures as proposed in the paper 
        kernel_dia =np.array([[0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0], 
                        [0,0,1,0,0,0,0,0,0], 
                        [0,0,0,1,0,0,0,0,0], 
                        [0,0,0,0,1,0,0,0,0], 
                        [0,0,0,0,0,1,0,0,0], 
                        [0,0,0,0,0,0,1,0,0], 
                        [0,0,0,0,0,0,0,1,0], 
                        [0,0,0,0,0,0,0,0,0]],np.uint8)
        kernel_row = np.array([0,1,1,1,1,1,1,1,1,1,1,1,0],np.uint8)
        kernel_col = np.array([[0],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [1],
                                [0]],np.uint8)  
        kernel_sqr =np.array([[0,0,0,0,0],
                        [0,1,1,1,0], 
                        [0,1,1,1,0], 
                        [0,1,1,1,0], 
                        [0,0,0,0,0]],np.uint8)               
        # Grayscale Morphological Closing Operations
        def morpho(img):
                closing_row = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_row)
                closing_dia = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_dia)
                closing_col = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_col)
                # Here implement max function 
                max_img=cv.max(closing_row, closing_dia,closing_col)
                # Calculate Generalized Greyscale Closing Image 'G'
                G = cv.absdiff(img, max_img)
                # Calculate Binary Hair Mask Pixel at the location (x,y), M(x,y)    
                ret, M = cv.threshold(G, 1, 255, cv.THRESH_BINARY_INV)
                return M
        def final_step(fi):
                dilate = cv.dilate(fi,kernel_sqr,iterations=1)
                #ret, M = cv.threshold(dilate, 1, 255, cv.THRESH_BINARY_INV)
                return dilate
        Mr=morpho(r_gray)
        Mb=morpho(b_gray)
        Mg=morpho(g_gray)
        M_glob = Mr+Mb+Mg  # This is the final binary Hair image
        # Check each "hair pixel" in image img_M. Make 4 line. Take shortest one. Identify 2 non hair pixel from 2 sides of line.
        # Send those pixel location for original image to replace respective non hair pixel intensities
        def bilinear_interpolation(img_M, original_im):
                rows, cols = img_M.shape
                x= 10
                y= 10
                rows_ =rows-10
                cols_ =cols-10
                for x in range(10,rows_): 
                        for y in range(10,cols_):
                                a,b=x,y
                                t1,t2,t3,t4=y,y,y,y
                                t11,t22,t33,t44=y,y,y,y
                                b5,a5=y+5,x+5
                                b_5,a_5=y-5,x-5
                                a_,b_=x-10,y-10
                                a10,b10=a+10,b+10
                                a20,b20=a+20,b+20
                                a__,b__=a-20,b-20
                                if img_M[x,y] < 10: # If this pixel is hair, find line lenghts
                                        for z in range(b,b10): # right direction search
                                                if img_M[x][z]>0:
                                                        right_len=z-b
                                                        break                                                 
                                                else:
                                                        right_len=10
                                        for z in range(b,b_,-1): # left direction search 
                                                if img_M[x][z]>0:
                                                        left_len=b-z
                                                        break
                                                else:
                                                        left_len=10
                                        for z in range(a,a10): # down direction search
                                                if img_M[z][y]>0:
                                                        down_len=z-a
                                                        break
                                                else:
                                                        down_len=10
                                        for z in range(a,a_,-1): # up direction search
                                                if img_M[z][y]>0:
                                                        up_len=a-z
                                                        break
                                                else:
                                                        up_len=10
                                        for z in range(a,a_,-1): # up right corner
                                                if img_M[z][t1]>0:
                                                        up_right=a-z
                                                        break                                                 
                                                else:
                                                        up_right=10
                                                t1 +=1
                                        for z in range(a,a_,-1): # up left corner
                                                if img_M[z][t2]>0:
                                                        up_left=a-z
                                                        break                                                 
                                                else:
                                                        up_left=10
                                                t2 -=1
                                        for z in range(a,a10): # down left corner
                                                if img_M[z][t3]>0:
                                                        down_left=z-a
                                                        break                                                 
                                                else:
                                                        down_left=10
                                                t3 -=1
                                        for z in range(a,a10): # down left corner
                                                if img_M[z][t4]>0:
                                                        down_right=z-a
                                                        break                                                 
                                                else:
                                                        down_right=10
                                                t4 +=1
                                        
                                ##find Horizontal or Vertical line
                                        V= down_len + up_len
                                        H=right_len + left_len
                                        DR = up_right + down_left
                                        DL = up_left + down_right
                                        min_lenght = min(V,H,DR,DL)
                                # Search from shortest line. Find 2 closest non hair pixel location and send it for calculation
                                        if min_lenght == H:      #HORIZONTAL   
                                                for z in range(b5,b10): # right direction search
                                                        if img_M[x][z]>0:   # if you can find 
                                                                x_right_nonhair=x
                                                                y_right_nonhair=z
                                                                break                                                 
                                                        else:
                                                                x_right_nonhair=a
                                                                y_right_nonhair=b10
                                                for z in range(b_5,b_,-1):
                                                        if img_M[x][z]>0:   # if you can find 
                                                                x_left_nonhair=x
                                                                y_left_nonhair=z
                                                                break                                                 
                                                        else:
                                                                x_left_nonhair=a
                                                                y_left_nonhair=b_
                                                calculate_new_image(original_im,x,y,x_right_nonhair,y_right_nonhair,x_left_nonhair,y_left_nonhair)
                                                continue
                                        elif  min_lenght == V:          # VERTICAL
                                                for z in range(a5,a10): # down direction search
                                                        if img_M[z][y]>0:
                                                                x_down_nonhair=z
                                                                y_down_nonhair=y
                                                                break
                                                        else:
                                                                x_down_nonhair=a10
                                                                y_down_nonhair=y
                                                for z in range(a_5,a_,-1): # up direction search
                                                        if img_M[z][y]>0:
                                                                x_up_nonhair=z
                                                                y_up_nonhair=y
                                                                break
                                                        else:
                                                                x_up_nonhair=a_
                                                                y_up_nonhair=y
                                                calculate_new_image(original_im,x,y,x_down_nonhair,y_down_nonhair,x_up_nonhair,y_up_nonhair)
                                                continue
                                        elif  min_lenght == DR:
                                                for z in range(a,a_,-1): # up right corner
                                                        if img_M[z][t11]>0:
                                                                x_up_right=z
                                                                y_up_right=t11
                                                                break                                                 
                                                        else:
                                                                x_up_right=a_
                                                                y_up_right=t11
                                                        t11 +=1
                                                for z in range(a,a10): # down left corner
                                                        if img_M[z][t22]>0:
                                                                x_down_left=z
                                                                y_down_left=t22
                                                                break                                                 
                                                        else:
                                                                x_down_left=a10
                                                                y_down_left=t22
                                                        t22 -=1
                                                calculate_new_image(original_im,x,y,x_up_right,y_up_right,x_down_left,y_down_left)
                                                continue
                                        else: # diagonal left
                                                for z in range(a,a_,-1): # up left corner
                                                        if img_M[z][t33]>0:
                                                                x_up_left=z
                                                                y_up_left=t33
                                                                break                                                 
                                                        else:
                                                                x_up_left=a_
                                                                y_up_left=t33
                                                        t33 -=1
                                                for z in range(a,a10): # down right corner
                                                        if img_M[z][t44]>0:
                                                                x_down_right=z
                                                                y_down_right=t44
                                                                break                                                 
                                                        else:
                                                                x_down_right=a10
                                                                y_down_right=t44
                                                        t44 +=1
                                                calculate_new_image(original_im,x,y,x_up_left,y_up_left,x_down_right,y_down_right)
        # This is where we should change hair pixel intensity with 2 closest non hair pixel intensity    
        def calculate_new_image(original,img_x,img_y,nonh1_x,nonh1_y,nonh2_x,nonh2_y):
                A=(nonh1_x-img_x)*(nonh1_x-img_x)
                B=(nonh1_y-img_y)*(nonh1_y-img_y)
                D1=math.sqrt(A+B)
                C=(nonh2_x-nonh1_x)*(nonh2_x-nonh1_x)
                D=(nonh2_y-nonh1_y)*(nonh2_y-nonh1_y)
                D2=math.sqrt(C+D)
                E=(nonh2_x-img_x)*(nonh2_x-img_x)
                F=(nonh2_y-img_y)*(nonh2_y-img_y)
                D3=math.sqrt(E+F)
                final_img[img_x,img_y]=final_img[nonh2_x,nonh2_y]*(D1/D2) + final_img[nonh1_x,nonh1_y]*(D3/D2)
                final_img

        ### This is where we call important calculation functions
        bilinear_interpolation(M_glob,image)   ## This results in hair removed image
        filter_final=final_step(final_img)     ## This filters/smooths final image
        ### We should write down new images one by one into new file or on top of it
        #cv.imwrite('final_dull.jpg',filter_final)

        return filter_final

#plt.subplot(122), plt.imshow(filter_final)
#plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()
