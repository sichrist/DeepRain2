#!/home/simon/anaconda3/bin/python

import sys
from imgprocessing import sequentialLabeling,convex_hull
from Dataset import DataProvider
import cv2 as cv
import numpy as np
from tracker import Tracker


windowname = 'OpenCvFrame'
cv.namedWindow(windowname)
cv.moveWindow(windowname,2600,40)


datafolder = "../PNG_NEW/MonthPNGData/YW2017.002_200806/"

data = DataProvider(datafolder)

def convexHull_(labels):
    hull = []
    for l,s,pts in labels:
        if s < 10:
            continue
        pts = [[x,y] for x,y in zip(pts[0],pts[1])]
        hull.append(convex_hull(pts))
    return hull

def optic_flow_cv(img1,img2):
    flow = cv.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 20, 3, 5, 1.2, 0)
    """
    flow = cv.calcOpticalFlowFarneback(img1, img2, 
                                                  #None,
                                                  #prevPts, 
                                                  None,
                                                  pyr_scale = 0.5, 
                                                  levels = 5, 
                                                  #winsize = 11, 
                                                  winsize = 5, 
                                                  iterations = 5, 
                                                  poly_n = 5, 
                                                  poly_sigma = 1.1,
                                                  flags=0) 
    """

    def draw_flow(imgs,flow, step=5):

        h, w = imgs.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
                
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(imgs, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, (255, 255, 255))
        return vis

    visualised = draw_flow(img1.copy(),flow)
    return visualised

tracker = Tracker(max_dist=7)

imglist = []
data.max_contrast()
data.binary()
for img in data:
    #labels = sequentialLabeling(img,max_dist=10,threshold=2)
    #hull = convexHull_(labels)
    #for h in hull:
    #    img[h] = 255


    imglist.append(img)
    if len(imglist) < 2:
        continue
    """
    visualised = optic_flow_cv(imglist[0],imglist[1])

    visualised = np.concatenate( (cv.cvtColor(img, cv.COLOR_GRAY2RGB),visualised), axis = 1 )

    imglist.pop(0)

    cv.imshow(windowname,visualised)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break
    """
    if cv.waitKey(25) & 0XFF == ord('q'):
        break
    img_cpy = cv.cvtColor(imglist[0].copy(),cv.COLOR_GRAY2RGB)
    clouds = tracker.calcFlow_clouds(imglist[0],imglist[1])
    for cloud in clouds:
      if cloud.size < 50:
        continue
      img_cpy = cloud.draw_hull(img_cpy)
      img_cpy = cloud.draw_path(img_cpy)

    cv.imshow(windowname,img_cpy)

    imglist.pop(0)
