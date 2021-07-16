# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:18:00 2021

@author: user
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from tensorflow.keras.preprocessing import image
    
def undesired_objects (image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey()
    
    return img2

#def find_mid(gray_image):
    

path = "E:/tech/ncdr/ncdr_image_extract/weather_image/scc201606090000.jpg"

# 讀取圖檔
img = cv2.imread(path)
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([15,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([160,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

lower_blue = np.array([110,50,50])
upper_blue = np.array([140,255,255])
mask3 = cv2.inRange(img_hsv, lower_blue, upper_blue)

# join my masks
mask = mask0+mask1+mask3

# set my output img to zero everywhere except my mask
output_img = img.copy()
output_img[np.where(mask==0)] = 0
 
# or your HSV image, which I *believe* is what you want
output_hsv = img_hsv.copy()
output_hsv[np.where(mask==0)] = 0

cv2.imshow('result', output_img)
cv2.waitKey()

img_gray=cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', img_gray)
cv2.waitKey()

kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(img_gray, kernel, iterations = 1)

cv2.imshow('canny', dilation)
cv2.waitKey()

biggest_component = undesired_objects(dilation)
extract_img = img.copy()
extract_img[np.where(biggest_component==0)] = 0

cv2.imshow('extract', extract_img)
cv2.waitKey()

#gray_biggest_component = cv2.cvtColor(biggest_component, cv2.COLOR_RGB2GRAY)
#thresh = cv2.threshold(biggest_component, 200, 255, cv2.THRESH_BINARY)[1]

#cv2.imshow('thresh_biggest_component', thresh)
#cv2.waitKey()

contours,_ = cv2.findContours(biggest_component.copy().astype(np.uint8), 1, 1) # not copying here will throw an error
#contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
#    cv2.CHAIN_APPROX_SIMPLE)
rect = cv2.minAreaRect(contours[0]) # basically you can feed this rect into your classifier
(x,y),(w,h), a = rect # a - angle

box = cv2.boxPoints(rect)
box = np.int0(box) #turn into ints
rect2 = cv2.drawContours(extract_img.copy(),[box],0,(0,0,255),5)

x_list = np.transpose(box)[0]
y_list = np.transpose(box)[1]

x_mid = (np.max(x_list) + np.min(x_list))/2
y_mid = (np.max(y_list) + np.min(y_list))/2

text = '(' + str(x_mid) + ', ' + str(y_mid) + ')'

cv2.putText(rect2, text, (int(x_mid), int(y_mid)), cv2.FONT_HERSHEY_TRIPLEX,
  1, (0, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('bounding', rect2)
cv2.waitKey()
#plt.imshow(rect2)
#plt.show()

from numpy import interp

lonRange = [90, 160] # flipped from descending to ascending
latRange = [10,60]

# the range of y and x pixels
yRange = [0, rect2.shape[0]]
xRange = [0, rect2.shape[1]]

xPixel = x_mid
yPixel = y_mid

lat = latRange[1] - interp(yPixel, yRange, latRange) # flipped again
lon = interp(xPixel, xRange, lonRange)

origin_cmp = cv2.drawContours(img.copy(),[box],0,(0,0,255),5)

text2 = '(' + str(format(lon,'.2f')) + ', ' + str(format(lat,'.2f')) + ')'

cv2.putText(origin_cmp, text2, (int(x_mid), int(y_mid)), cv2.FONT_HERSHEY_TRIPLEX,
  1, (0, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('origin_cmp', origin_cmp)
cv2.waitKey()

cv2.destroyAllWindows()

# 寫入圖檔
out_path = "extract.jpg"
cv2.imwrite(out_path, extract_img)