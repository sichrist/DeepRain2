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
#cv.namedWindow(windowname)
#cv.moveWindow(windowname,2600,40)


class Tracker(object):
    """

    docstring for Tracker

    """
    def __init__(self, path=None,max_dist = 20,binary=True,max_contrast=True,transform=None,threshold=5):
        super(Tracker, self).__init__()

        if path is not None:
            self.path = path
            self.data = DataProvider(path)

            if max_contrast:
                self.data.max_contrast()
            if binary:
                self.data.binary()
        self.max_dist = max_dist
        self.threshold = threshold
    
    def label_To_index(self,label,img):

        return np.where(img == label)

    def getClouds(self,cloudlist,img):
        clouds = [Cloud(idx) for label,size,idx in cloudlist]
        return clouds

    def sequentialLabeling(self,img):

        """
    
    
            Label clouds and return array of tuble (label,cloudsize)
            labels are sorted in descending order by cloudsize
            Explicit location of cloud labeld by label_A 
            can be found by np.where(img == label_A)


        """
        img = img.copy().astype(np.uint32)
        img[img >= self.threshold] = 1

        true_value = img.max()
        x,y = np.where(img == 1)


        collision = dict()
        label = 2

        for i,j in zip(x,y):
            i_X = slice(i-self.max_dist,i+self.max_dist)
            j_Y = slice(j-self.max_dist,j+self.max_dist)

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

    def mapToColor(self,img,cloud_size):
        COL = MplColorHelper('hsv', 0, len(cloud_size))

        colors = {}
        for i,elem in enumerate(cloud_size):
            colors[elem[1]] = COL.get_rgb(i)
        img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)

        for label,size in cloud_size:
            index = np.where(img[:,:,0] == label)
            img[index] = colors[size]

        return img

    def draw_flow(self,imgs,flow, step=6):

                h, w = imgs.shape[:2]
                y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
                fx, fy = flow[y, x].T
                lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
                
                lines = np.int32(lines + 0.5)
                vis = cv.cvtColor(imgs, cv.COLOR_GRAY2BGR)
                cv.polylines(vis, lines, 0, (255, 255, 255))
                #for (x1, y1), (x2, y2) in lines:
                #    cv.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
                return vis

    def calcFlow(self,img0,img1,prevPts=None,nextPts=None):
        
        flow = cv.calcOpticalFlowFarneback(img0, img1, 
                                              #None,
                                              #prevPts, 
                                              nextPts,
                                              pyr_scale = 0.5, 
                                              levels = 5, 
                                              #winsize = 11, 
                                              winsize = 5, 
                                              iterations = 5, 
                                              poly_n = 5, 
                                              poly_sigma = 1.1,
                                              flags=0) 
                                              #flags = cv.OPTFLOW_USE_INITIAL_FLOW)

        return flow

    def showset(self):
        def show(img):
            cv.imshow(windowname,img)
        
        for i,img in enumerate(self.data):
            cloudlist = self.sequentialLabeling(img)
            clouds = self.getClouds(cloudlist,img)
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            for c in clouds:
                #img = c.bbox(img)
                
                img = c.draw_hull( img )
            show(img)
            if cv.waitKey(25) & 0XFF == ord('q'):
                break
            #img = self.mapToColor(img,cloudsize)

            


        cv.destroyAllWindows()

    def create(self,inputPath, outputPath, delay, finalDelay, loop):
            cmd = "convert -delay {} {}*.png -delay {} -loop {} {}".format(
            delay, inputPath, finalDelay, loop,
            outputPath)
            print(cmd)
            os.system(cmd)

    def calcFlow_clouds(self,img1,img2):

        """
            
            img1, img2 needs to be binary + maxcontrast

        """


        def average_movement(flow,clouds):

                """

                    Average over all directions calculated by optic flow per cloud

                """

  
                tmp = np.zeros(flow.shape)

                for i in range(len(clouds)):
                
                    avg_len = np.sqrt( (flow[clouds[i].points]**2).sum(axis=1) ).sum(axis=0) / len(clouds[i].points[0])
                    avg_direction = flow[clouds[i].points].sum(axis=0)  / len(clouds[i].points[0])
                    avg_len *= 1
                    avg_dir = (avg_direction**2).sum(0)
                    if avg_dir == 0:
                        continue
                    avg_direction = avg_direction * (avg_len / np.sqrt( avg_dir ) )

                    #avg_direction = [avg_direction[1],avg_direction[0]]
                    clouds[i].set_direction(avg_direction)
                    tmp[clouds[i].points] = avg_direction
                flow = tmp

                return flow

        def median_movement(flow,clouds):

            def pol2cart(rho, phi):
                """
                    stolen from Stackoverflow

                    https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates

                """
                x = rho * np.cos(phi)
                y = rho * np.sin(phi)
                return(x, y)

            def cart2pol(x, y):
                """
                    stolen from Stackoverflow

                    https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
                      
                """
                rho = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return(rho, phi)

            tmp = np.zeros(flow.shape)
            for i in range(len(clouds)):
                x,y = clouds[i].points
                rho,phi = cart2pol(x,y)
                carthesian = [[r,p] for r,p in zip(rho,phi)]

                median = sorted(carthesian, key=lambda tup:tup[1])
                median = median[len(median) // 2]

                avg_direction = pol2cart(median[0],median[1])

                clouds[i].set_direction(avg_direction)

            return flow

        def average_movement_normalized(flow,clouds):

            def pol2cart(rho, phi):
                """
                    stolen from Stackoverflow

                    https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates

                """
                x = rho * np.cos(phi)
                y = rho * np.sin(phi)
                return(x, y)

            def cart2pol(x, y):
                """
                    stolen from Stackoverflow

                    https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
                      
                """
                rho = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return(rho, phi)


            tmp = np.zeros(flow.shape)
            for i in range(len(clouds)):
                x,y = clouds[i].points
                rho,phi = cart2pol(x,y)

                carthesian = [[r,p] for r,p in zip(rho,phi)]
                
                carthesian = np.mean(np.array(carthesian),axis=0)                
                avg_direction = pol2cart(carthesian[0],carthesian[1])

                clouds[i].set_direction(avg_direction)


            return flow


        flow = self.calcFlow(img1,img2,None)

        cloudlist1 = self.sequentialLabeling(img1)
        cloudlist2 = self.sequentialLabeling(img2)
        clouds1 = self.getClouds(cloudlist1,img1)
        clouds2 = self.getClouds(cloudlist2,img2)

        flow = average_movement(flow,clouds1)
        #flow = median_movement(flow,clouds1)
        #flow = average_movement_normalized(flow,clouds1)

        """
        mask = np.zeros_like(img1)
        vis = self.draw_flow(mask,flow)
        img1 = cv.cvtColor(img1,cv.COLOR_GRAY2RGB)
        img2 = cv.cvtColor(img2,cv.COLOR_GRAY2RGB)

        frame = np.concatenate((img1,img2),axis=1)
        frame = np.concatenate((frame,vis),axis=1)

        h,w = frame.shape[:2]
        h1,w1 = img1.shape[:2]
        
        frame[:,w1-1:w1+1,:] = [255,255,255]
        scale = 0.5
        """
        """
        frame = cv.resize(frame,(int(w * scale),int(h * scale)))
        while True:
            cv.imshow(windowname,frame)

            if cv.waitKey(25) & 0XFF == ord('q'):
                break
        """
        #cv.imshow("BOING",vis)
        return clouds1




    """
     ---------- Everything below this line is for testing --------------

    """



    def showFlow(self,create_gif=False,name="clouds.gif",nbr_imgs=0):
        img_list = []
        flow = None 
        scale = 0.6
        if create_gif: 
            folder = "GIF/"
            if not os.path.exists(folder):
                os.mkdir(folder)

        for i,img in enumerate(self.data):
            if len(img_list) == 0:
                mask = np.zeros_like(img)

            h,w = img.shape[:2]
            #img = cv.resize(img,(int(w * scale),int(h * scale)))
            #img = cv.resize(img,(w,h))
            #cv.imshow(windowname,img)
            #if cv.waitKey(25) & 0XFF == ord('q'):
            #    break
            #continue
            cloudlist = self.sequentialLabeling(img)
            clouds = self.getClouds(cloudlist,img)

            #clouds as points for tracking
            #pts = [ cloud.points for cloud in clouds]
            #center of mass as points for tracking
            pts = [ cloud.center_of_mass for cloud in clouds]

            # need to reshape pts
            pts = np.array(pts)
            x,y = pts.shape
            pts = pts.reshape(y,1,x) 
            
            if len(img_list) >= 2:
                img_list.pop(0)
            
            img_list.append((img,pts) )

            if len(img_list) < 2:
                continue
            
            flow = self.calcFlow(img_list[0][0],img_list[1][0],pts)

            #print()


            """
                TEST
                
                Set everything to zero except biggest cloud
            """


            def average_movement(flow):
                #flow = flow.astype(np.float32)
                tmp = np.zeros(flow.shape)
                for i in range(len(clouds)):
                
                    avg_len = np.sqrt( (flow[clouds[i].points]**2).sum(axis=1) ).sum(axis=0) / len(clouds[i].points[0])
                    avg_direction = flow[clouds[i].points].sum(axis=0)  / len(clouds[i].points[0])
                    avg_len *= 16
                    avg_dir = (avg_direction**2).sum(0)
                    if avg_dir == 0:
                        print("DIR SMALL:",flow[clouds[i].points][:,0],flow[clouds[i].points][:,1],avg_dir)
                        continue
                    print(avg_direction)
                    avg_direction = avg_direction * (avg_len / np.sqrt( avg_dir ) )

                    tmp[clouds[i].points] = avg_direction
                flow = tmp

                return flow
            flow = average_movement(flow)

            vis = self.draw_flow(mask,flow)
            
            #mask = cv.cvtColor(vis,cv.COLOR_RGB2GRAY)




            #print("MAX:",img_list[0][0].max())
            #frame = np.concatenate((img_list[0][0],mask),axis=1)
            frame = cv.cvtColor(img_list[0][0],cv.COLOR_GRAY2RGB)
            for c in clouds:
                frame = c.paintcolor(frame)
            frame = np.concatenate((frame,vis),axis=1)

            
            
            cv.imshow(windowname,frame)
            print(i,end="\r")


            if cv.waitKey(25) & 0XFF == ord('q'):
                break


            if create_gif and len(img_list) == 2:
                filename = "{0:0>5}".format(i)
                cv.imwrite(os.path.join(folder,filename+".png"),frame)
                if i == nbr_imgs and nbr_imgs != 0:
                    break
        if create_gif:
            self.create(folder,name,20,250,0)



"""
t = Tracker("../PNG",max_dist=1)
#t.showset()
#t.showFlow(create_gif=False,name="clouds_as_center_of_mass.gif")

img1 = t.data[0]
img2 = t.data[1]

print(img1.shape)
print(img2.shape)

print(np.where(img1 > 1))

clouds = t.calcFlow_clouds(img1,img2)

img1 = cv.cvtColor(img1, cv.COLOR_GRAY2RGB)


for cloud in clouds:
    img1 = cloud.draw_hull( img1 )
    img1 = cloud.draw_path(img1)

while True:
    cv.imshow(windowname,img1)
    if cv.waitKey(25) & 0XFF == ord('q'):
        break   
cv.destroyAllWindows()



def predict(clouds,start_i,data,point_to_predict):
    
    a = False
    while not a:
        
        for cloud in clouds:
            print("HIER")
            a = cloud.is_inPath(point_to_predict)
            if a:
                break
        #ptp[0] += 1
        #if ptp[0] > 3000:
        #    print("OUT OF LIMIT")
        #    break
    #print("WHAT",a)


exit(0)
ptp = [338,690]
predict(clouds,0,t.data,ptp)

"""

def full():
    folder = "GIF/"
    if not os.path.exists(folder):
        os.mkdir(folder)



    img_old = t.data[0]
    kernel = np.ones((3,3),np.uint8)
    for i in range(1,len(t.data),3):
        img = t.data[i]
        #img = cv.erode(img,kernel,iterations = 2)
        img = cv.dilate(img,kernel,iterations = 1)
        
        clouds = t.calcFlow_clouds(img_old,img)

        img_old = cv.cvtColor(img_old, cv.COLOR_GRAY2RGB)
        for j,cloud in enumerate(clouds):
            #if cloud.size < 10:
                #continue
            #img_old = cloud.paintcolor(img_old)
            img_old = cloud.draw_hull( img_old )
            img_old = cloud.draw_path( img_old )

            if j > 10:
                break

        cv.imshow(windowname,img_old)
        filename = "{0:0>5}".format(i)
        print(filename)
        cv.imwrite(os.path.join(folder,filename+".png"),img_old)

        if cv.waitKey(25) & 0XFF == ord('q'):
            break

        img_old = img
    cv.destroyAllWindows()
    name = "path_direction_max2.gif"
    t.create(folder,name,50,250,0)


def show_label():

    for img in t.data:
        img_copy = img.copy()
        img = img.astype(np.uint32)
        cloudlist = t.sequentialLabeling(img)
        clouds = t.getClouds(cloudlist,img)
        img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2RGB)
        for c in clouds:
            img_copy = c.draw_hull( img_copy )

        cv.imshow(windowname,img_copy)
        if cv.waitKey(25) & 0XFF == ord('q'):
            break   
    cv.destroyAllWindows()


def oneCalc():
    pattern = ".*"
    datafolder = "../PNG_NEW/MonthPNGData/YW2017.002_200806/"
    t = Tracker(datafolder,pattern=pattern,max_dist=1)
    img_old = t.data[0]
    img_new = t.data[9]

    clouds = t.calcFlow_clouds(img_old,img_new)

    for i,img in enumerate(t.data):
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        print("FRAME: ",i,end="\r")
        for cloud in clouds:
            if cloud.size < 200:
                continue
            img = cloud.draw_hull( img )
            img = cloud.draw_path( img )

        cv.imshow(windowname,img)
        if cv.waitKey(25) & 0XFF == ord('q'):
            break   
    cv.destroyAllWindows()

#oneCalc()
#show_label()
#full()