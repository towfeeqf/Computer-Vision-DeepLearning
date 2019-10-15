# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:47:23 2019

@author: SonyTF
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 9,6

def canny(image):
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # to change to rgb
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def dislay_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 =line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            #print(line)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
            [(200,height),(1000,height),(550,250)]
            ])
    mask= np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255) # fill the polygon of size mask with triangle and color 255 i.e white
    masked_image = cv2.bitwise_and(image,mask)
    
    return masked_image

# hough transform 2
    


image = cv2.imread('test_image.jpg')
lane_image =np.copy(image)

canny = canny(lane_image)


cropped_image= region_of_interest(canny)

lines = cv2.HoughLinesP(cropped_image,2, np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)# hough accumulator array, 2 pixels

line_image = dislay_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image, 1,1)


cv2.imshow('result',combo_image)
cv2.waitKey(0)


#plt.imshow(cropped_image,cmap='gray')
#plt.show()

    # The lines below all goes into the canny function
    
    ## opencv reads BGR so we need to convert to RGB first
    #lane_image  = cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    #
    ##step3: Gaussian blur ( kernel 5,5  , deviation :0)
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #
    ##step4: apply a canny method does apply a 5,5 kernel method inside it
    ## cv2.Canny(image, low_threshold, high_threshold)
    #canny = cv2.Canny(blur,50,150)

#step5 : region of interest



#cv2.imshow('result',canny)
#cv2.waitKey(0)
