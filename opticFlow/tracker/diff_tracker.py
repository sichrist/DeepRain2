#!/home/simon/anaconda3/bin/python

import sys
print(sys.executable)
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
import cv2 as cv
from Dataset import DataProvider
from Cloud import Cloud
from ColorHelper import MplColorHelper
from time import sleep 

windowname = 'OpenCvFrame'
cv.namedWindow(windowname)
cv.moveWindow(windowname,2600,40)




def sequentialLabeling(img,threshold=1,max_dist=1):
        print("sequentialLabeling")
        """
    
    
            Label clouds and return array of tuble (label,cloudsize)
            labels are sorted in descending order by cloudsize
            Explicit location of cloud labeld by label_A 
            can be found by np.where(img == label_A)


        """
        img = img.copy().astype(np.uint32)
        img[img >= threshold] = 1

        true_value = img.max()
        x,y = np.where(img == 1)

        collision = dict()
        label = 2

        for i,j in zip(x,y):
            print("label: ",label,end="\r")
            i_X = slice(i-max_dist,i+max_dist)
            j_Y = slice(j-max_dist,j+max_dist)

            window = img[i_X,j_Y]

            neighbours = np.argwhere(window > 1)


            if len(neighbours) == 0:
                window[window == 1] = label
                label +=1
                img[i_X,j_Y] = window

            elif len(neighbours) == 1:
                window[window == true_value] = window[neighbours[0,0],neighbours[0,1]]
                img[i_X,j_Y] = window


            # handle label collisions

            else:
                k = np.amax(window)
                img[i,j] = k
                for index in neighbours:
                    nj = window[index[0], index[1]]

                    if nj != k:
                        if k not in collision:
                            collision[k] = set()
                        collision[k].add(nj)
                        if collision[k] is None:
                            del collision[k]



        def changeLabel(elem):
            c_label = collision[elem]
            for l in c_label:
                img[img == l] = elem


        def rearangeCollisions():
            for elem in collision:
                for item in collision[elem]:
                    if item in collision:
                        collision[elem] = (collision[elem] | collision[item])
                        collision[item] = set()
                if elem in collision[elem]:
                    collision[elem].remove(elem)


        rearangeCollisions()


        for i,elem in enumerate(collision):
            if collision[elem] is None:
                continue
            changeLabel(elem)

        cloud_size = []

        for i in range(2,label):
            idx = np.where(img == i)
            a = len(idx[0])

            if a == 0:
                continue
            cloud_size.append((i,a,idx))
        cloud_size = sorted(cloud_size, key=lambda x: x[1],reverse = True)

        #img2 = img.copy()
        #img2 = img2 * (255 /img2.max())
        #cv.imshow("LABELING",img2.astype(np.uint8))
        #cv.moveWindow("LABELING",0,40)
        print("NBR LABEL: ",label)
        return cloud_size










data = DataProvider("../PNG")
data.max_contrast()
data.binary()

kernel = np.ones((3,3),np.uint8)
for img in data:
    img = cv.dilate(img,kernel,iterations = 1)
    #img = cv.erode(img,kernel,iterations = 2)
    cloudlist = sequentialLabeling(img,max_dist=1)
    clouds = [Cloud(idx) for label,size,idx in cloudlist]
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    for cloud in clouds:
        img[cloud.points] = [255,255,255]
        img = cloud.draw_hull(img)

    cv.imshow(windowname,img)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break