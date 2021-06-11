# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 12:18:00 2021

@author: user
"""
import configparser
config = configparser.ConfigParser()    # 注意大小寫
config.read("config.ini")   # 配置檔案的路徑

weather_path = config['main']['origin_path']
extract_path = config['main']['destination_path']
suffix = config['main']['suffix']
image_type = config['main']['image_type']


import os

work_path = os.getcwd()

def mkdir(create_path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(create_path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(create_path)
        print('-----建立成功-----')

    else:
        #如果目錄已存在，則不建立，提示目錄已存在
        print(create_path+'目錄已存在')

import numpy as np
import cv2

mkdir(extract_path)

    
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
    #cv2.imshow("Biggest component", img2)
    #cv2.waitKey()
    
    return img2

for file in os.listdir(weather_path):
    #path = "scc201606090000.jpg"
    dir_path = os.path.abspath(weather_path)
    path = dir_path + "\\" + file
    print(path)
    
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
    
    #cv2.imshow('result', output_img)
    #cv2.waitKey()
    
    img_gray=cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow('gray', img_gray)
    #cv2.waitKey()
    
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(img_gray, kernel, iterations = 1)
    
    #cv2.imshow('canny', dilation)
    #cv2.waitKey()
    
    biggest_component = undesired_objects(dilation)
    extract_img = img.copy()
    extract_img[np.where(biggest_component==0)] = 0
    
    #cv2.imshow('extract', extract_img)
    #cv2.waitKey()
    
    cv2.destroyAllWindows()
    
    # 寫入圖檔
    out_path = extract_path + '\\' + file.split('.')[0] + suffix +'.' +image_type
    print(out_path)
    cv2.imwrite(out_path, extract_img)