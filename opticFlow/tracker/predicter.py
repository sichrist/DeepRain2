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
from tracker import Tracker
import re
regex = r"scaled_([0-9]+).png"
windowname = " BOINg "


y,x = 450,550
data = DataProvider("../PNG",openIMG=False)
tracker = Tracker(max_dist=2)
data.max_contrast()
data.binary()

def showPoint(data):

    for img in data:
    #img = data[0]
        img = data._openIMG(img)
        img = cv.circle(img, 
                    (x,y), 
                    5, 
                    [255], 
                    thickness=1, 
                    lineType=8, 
                    shift=0) 
        cv.imshow("YOLO",img)
        if cv.waitKey(25) & 0XFF == ord('q'):
            break   


def get_img_infos(data,point):
    files_info = []
    rain_img = 0
    x,y = point


    for filename in data:
        matches = re.search(regex, filename)
        date = None
        rain = False
        
        if matches:
            date = int(matches[1])
        else:
            continue

        img = data._openIMG(filename)

        if img[y,x] > 0:
            rain = True
            rain_img += 1

        files_info.append((filename,rain,date))


    return files_info


def predict(fileinfos,tracker,ttp=30,steps=5):
    start = int(ttp / steps) + 1
    

    correct = 0

    for i in range(start,len(fileinfos)):


        if fileinfos[i][2] == False:
            continue

        img_to_pred,label,timestamp = fileinfos[i]
        # 30 mins prior
        img_past = fileinfos[i-start][0]
        # 35 mins prior
        img_past_mo = fileinfos[i-start-1][0]

                        
        img2 = data._openIMG(img_past_mo)
        img1 = data._openIMG(img_past)
        img = np.array(Image.open(img_to_pred))

        clouds = tracker.calcFlow_clouds(img2,img1)


        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        img = cv.circle(img, 
                    (x,y), 
                    5, 
                    [255,0,0], 
                    thickness=1, 
                    lineType=8, 
                    shift=0) 

        for cloud in clouds:

            img = cloud.draw_hull(img)
            #if cloud.size < 100:
            #    continue
            img = cloud.draw_path(img)
        
        cv.namedWindow(windowname)
        cv.moveWindow(windowname,2600,40)
        
        cv.imshow(windowname,img)
        while True:
            if cv.waitKey(25) & 0XFF == ord('q'):
                break
        




#showPoint(data)
fileinfos = get_img_infos(data,point=(x,y))
predict(fileinfos,tracker)

