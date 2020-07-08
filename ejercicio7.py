# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:03:20 2020

@author: miner
"""

#from tool._fixedInt import *
import tool._fixedInt as fp
from skimage.exposure import rescale_intensity
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2



# In[]:Functions

def swapKernel(kernel):
    kernel[:,[0, 2]] = kernel[:,[2, 0]]
    kernel[[0,2],:]  = kernel[[2,0],:]
   
    return kernel

def scaling(conv_image):
    conv_image = rescale_intensity(conv_image, in_range=(0, 255))
    return ((conv_image*255).astype("uint8"))


def scalingI(I,newMax,newMin):
    max_v = np.amax(I)
    min_v = np.amin(I)
    return (np.true_divide((I-min_v)*(newMax-newMin),(max_v-min_v)) + min_v)
    
def kernelNorm(kernel):
    max_v = np.amax(np.absolute(kernel))
    return (kernel/max_v)
    
    
def printSizes(rowIm,colIm,rowScalled,colScalled):
    
    print('filas imagen original = {}'.format(rowIm))
    print('colum imagen original = {}'.format(colIm))
    print('filas imagen escalada = {}'.format(rowScalled))
    print('colum imagen escalada = {}'.format(colScalled))
    print('-'*70)
    
def convolution2D (image, kernel):

    imrows, imcols = image.shape
    krows , kcols  = kernel.shape
    center    = int(krows-(krows+1)/2)
    imConv2D  = np.zeros((int(imrows),int(imcols)))
    
    for n in range (int(imrows)):
        for m in range (int(imcols)):
            element = 0        
            for i in range(int(krows)):
                for j in range(int(krows)):
                    if (n-center+i)>= 0 and (m-center+j)>=0 and (n-center+i)<imrows and (m-center+j)<imcols:
                        element += image[n-center+i,m-center+j]*kernel[i,j]            
            imConv2D[n,m]=element 
    return imConv2D

def plotHist(conv_image, convCV,name,fig):
    [x  ,y]   = np.unique(conv_image,return_counts=True)
    [xcv,ycv] = np.unique(convCV,return_counts=True)
    
    
    plt.figure(fig)
    plt.subplot(2,1,1)
    plt.stem(xcv,ycv,'ko',label='open CV reference',use_line_collection=True)
    plt.legend()
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.stem(x,y,'ko',label='image with {} '.format(name),use_line_collection=True)
    plt.legend()
    plt.grid()
    
    plt.show()
    

# In[1]:

kernels = {
            'smallBlur'   : np.ones((7, 7), dtype="float") * (1.0 / (7 * 7)),
            'largeBlur'   : np.ones((21, 21), dtype="float") * (1.0 / (21 * 21)),
            'sharpen'     : np.array(([ 0, -1,  0],
                                	  [-1,  5, -1],
                                	  [ 0, -1,  0]), dtype="int"),
            'laplacian'   :np.array(( [ 0,  1,  0],
                                	  [ 1, -4,  1],
                                	  [ 0,  1,  0]), dtype="int"),
            'edge_detect' :np.array(( [ 1,  0, -1],
                                	  [ 0,  0,  0],
                                	  [-1,  0,  1]), dtype="int"),
            'edge_detect2':np.array(( [-1, -1, -1],
                                      [-1,  8, -1],
                            	      [-1, -1, -1]), dtype="int"),
            'sobelX'      :np.array(( [-1,  0,  1],
                            	      [-2,  0,  2],
                            	      [-1,  0,  1]), dtype="int"),
            'sobelY'      : np.array(([-1, -2, -1],
                                	  [ 0,  0,  0],
                                      [ 1,  2,  1]), dtype="int") 
    }




ap = argparse.ArgumentParser( description="Convolution 2D: This function compare opencv and interative method.")

#reception of data

path = "descarga.jpg"
#path = "foto1.jpg"

ap.add_argument("-i", "--image", required=False, help="Path to the input image",default=path)
#ap.add_argument("-k", "--kernel",               help="Path to the kernel")
args  = ap.parse_args()
image = cv2.imread(args.image,1)
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


kernel = kernels['edge_detect2']

#swaps kernel
kernel = swapKernel(kernel)



#scaling 0  
opencvOutput = cv2.filter2D(gray, -1, kernel) 
print('OPENC CV SCALLED AND CONV IMAGE')

#scaling 1
conv1 = convolution2D(gray, kernel)
img   = scalingI(conv1,255,0).astype('uint8')

#scaling 2
In = scalingI(gray,1,0)
kn = kernelNorm(kernel)
conv = convolution2D(In, kn)                      
conv = equalize_adapthist(np.clip(conv,0,1))
conv_image = scalingI(conv,255,0).astype('uint8')


#show image
cv2.imshow("Edge Dectect - opencv", opencvOutput) 
cv2.imshow("jime #2 ", conv_image)
#cv2.imshow("jime #1 ", img)
plotHist(conv_image, opencvOutput,"scaling #2",1)
#plotHist(img, opencvOutput,"scaling #1",2)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
 

     