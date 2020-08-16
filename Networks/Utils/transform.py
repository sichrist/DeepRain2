import cv2 
from multiprocessing import Process, cpu_count
import os
from time import sleep
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf


class fromCategorical(object):
    """docstring for fromCategorical"""
    def __init__(self, conditions):
        super(fromCategorical, self).__init__()
        self.conditions = conditions
        

class ToCategorical(object):
    """

        Map array values to values in conditions
        
        [1,50,60]

        values between 1 and 50 will be mapped to index 0
        => [1,0,0]

    """
    def __init__(self, conditions):
        super(ToCategorical, self).__init__()
        self.conditions = np.array(conditions)
        self.numClasses = len(self.conditions) -1

    def __call__(self,array):
        newVector = np.zeros((*array.shape,self.numClasses))
        for i in range(1,self.numClasses):
            value = self.conditions[i]
            valuePrev = self.conditions[i-1]
            idx = np.where((array < value) & (array >=valuePrev))
            if len(idx[0]) == 0:
                continue

            classV = np.zeros((self.numClasses))
            classV[i-1] = 1

            
            newVector[idx] = classV

        return newVector

class Binarize(object):
    """docstring for Binarize"""
    def __init__(self,threshold=0,value=255):
        super(Binarize, self).__init__()
        self.threshold = threshold
        self.value = value


    def __call__(self,img):
        img[np.where(img > self.threshold)] = self.value
        img[np.where(img <= self.threshold)] = 0
        return img
class LinearMapping(object):
    """docstring for LinearMapping"""
    def __init__(self, classes = 10):
        super(LinearMapping, self).__init__()
        self.classes = classes
        self.factor = classes / 255
        
    def __call__(self,img):
        img = img * self.factor
        img = img.astype(np.uint8)
        return img.astype(np.float64) / self.classes
        

class Flatten(object):
    """docstring for Flatten"""
    def __init__(self):
        super(Flatten, self).__init__()       

    def __call__(self,img):
        img = img.flatten()
        return img      


class ToUint8(object):
    """docstring for ToUint8"""
    def __init__(self):
        super(ToUint8, self).__init__()
              
        
    def __call__(self,img):
        return (img * 255).astype(np.uint8)

class NormalDist(object):
    """docstring for Normaldistribution"""
    def __init__(self):
        super(NormalDist, self).__init__()
              
        
    def __call__(self,img):
        return img - 127.0


########################################################################
###                     PRETRANSFORMATIONS                           ###
########################################################################



class cutOut(object):
    """docstring for cutOut"""
    def __init__(self,slices):
        super(cutOut, self).__init__()
        #assert type(slices) is slice, "Parameter slices needs to be type of slice!"
        self.idx = slices
        self.slices = [slice(slices[0],slices[1]),slice(slices[2],slices[3])]

    def __call__(self,img):
        return img[self.slices]

    def __str__(self):
        savefolder=str(self.idx[0])+"x"+str(self.idx[1])+"_"+str(self.idx[0])+"x"+str(self.idx[1])
        return savefolder



class resize(object):
    """
    
        resizes image to dimension dim


    """
    def __init__(self, dim):
        super(resize, self).__init__()
        self.dim = dim

    def __call__(self,img):

        x,y = self.dim
        img = cv2.resize(img,(y,x))
        return img
        
    def __str__(self):
        savefolder = ""
        for i in self.dim:
            savefolder += str(i)+"x"
        savefolder = savefolder[:-1]
        return savefolder




class ImageToPatches(object):
    """docstring for ImageToPatches"""
    def __init__(self,outputsize,inputsize = None,stride=(0,0)):
        super(ImageToPatches, self).__init__()
        self.stride = stride
        self.outputsize = outputsize
        self.inputsize = inputsize


        if self.inputsize is None:
            self.inputsize = self.outputsize

        self.offset_x = (self.inputsize[0] - self.outputsize[0]) // 2
        self.offset_y = (self.inputsize[1] - self.outputsize[1]) // 2


    def get_y_by_index(self,y,i,j):

        
        patch_y  = np.zeros(self.outputsize,dtype=np.uint8)

        start_x = i * self.outputsize[0] - self.stride[0]
        end_x   = start_x + self.outputsize[0] 

        start_y = j * self.outputsize[1] - self.stride[1]
        end_y   = start_y + self.outputsize[1]


        start_x = start_x if start_x > 0 else 0
        start_y = start_y if start_y > 0 else 0
        
        end_x = end_x if end_x < self.outputsize[0] else self.outputsize[0]
        end_y = end_y if end_y < self.outputsize[1] else self.outputsize[1]


        



    def __call__(self,img):
        x = img
        y = img

        input_matrix  = [] 
        output_matrix = []

        
        for index_i in range(0,x.shape[0],self.outputsize[0]-self.stride[0]):
            patch_in = []
            patch_ou = []
            start_x = index_i - self.offset_x
            end_x   = index_i + self.outputsize[0] + offset_x
            start_x = start_x if start_x >= 0 else 0
            end_x = end_x if end_x <= x.shape[0] else x.shape[0]

            for index_j in range(0,y.shape[1],self.outputsize[1]-self.stride[1]):
                
                patch_x  = np.zeros(self.inputsize,dtype=np.uint8)
                patch_y  = np.zeros(self.outputsize,dtype=np.uint8)


                start_y = index_j - self.offset_y
                end_y   = index_j+self.outputsize[1] + self.offset_y

                start_y = start_y if start_y >= 0 else 0
                end_y = end_y if end_y <= x.shape[1] else x.shape[1]
                
                patchx = x[start_x:end_x,start_y:end_y]
                patchy = y[index_i:index_i+self.outputsize[0],index_j:index_j+self.outputsize[1]]

                patch_y[:patchy.shape[0],:patchy.shape[1]] = patchy
                
                start_x_in = 0
                end_x_in = patchx.shape[0]
                start_y_in = 0
                end_y_in = patchx.shape[1]
                if patchx.shape[0] != self.inputsize[0]:
                    diff = self.inputsize[0] - patchx.shape[0]
                    if start_x == 0:
                        start_x_in += diff
                        end_x_in += diff



                if patchx.shape[1] != self.inputsize[1]:
                    diff = self.inputsize[1] - patchx.shape[1]
                    if start_y == 0:
                        start_y_in += diff
                        end_y_in += diff




                
                patch_x[start_x_in:end_x_in,start_y_in:end_y_in] = patchx

                patch_in.append(patch_x)
                patch_ou.append(patch_y)

            input_matrix.append(patch_in)
            output_matrix.append(patch_ou)
        
        return np.array(input_matrix),np.array(output_matrix)
        

########################################################################
###                             WORKER                              ###
########################################################################


def fProcess(listOfFiles,savedir,transformations):

    while listOfFiles:
        file = listOfFiles.pop()

        filename = file.split('/')[-1]

        pathToWrite = os.path.join(savedir,filename)

        if os.path.exists(pathToWrite):
            print("File ",pathToWrite," exists",len(listOfFiles))
            continue

        img = np.array(Image.open(file))

        for transformation in transformations:
            img = transformation(img)

        img = Image.fromarray(img)
        img.save(pathToWrite)


def splitImgProcess(listOfFiles,savedir,transformation):
    inputfilename = str(transformation.inputsize[0])+"x"+str(transformation.inputsize[1])
    outputfilename = str(transformation.outputsize[0])+"x"+str(transformation.outputsize[1])

    while listOfFiles:
        file = listOfFiles.pop()
        filename = file.split('/')[-1]
        foldername = filename.split(".")[0]
        pathToWrite = os.path.join(savedir,foldername)

        if os.path.exists(pathToWrite):
            print("File ",pathToWrite," exists",len(listOfFiles))
            continue
        os.mkdir(pathToWrite)
        xfile = os.path.join(pathToWrite,inputfilename)
        yfile = os.path.join(pathToWrite,outputfilename)

        os.mkdir(xfile)
        os.mkdir(yfile)

        img = np.array(Image.open(file))
        x,y = transformation(img)
        x_x,x_y = x.shape[:2]
        y_x,y_y = y.shape[:2]

        for i in range(x_x):
            for j in range(x_y):
                file = os.path.join(xfile,str(i)+"_"+str(j)+".png")
                img = Image.fromarray(x[i,j])
                img.save(file)
                file = os.path.join(yfile,str(i)+"_"+str(j)+".png")
                img = Image.fromarray(y[i,j])
                img.save(file)




def transformImages(listOfFiles,transformation,savedir,saveListOfFiles,target=fProcess):
    """

        Transforms images specified by parameter transformation
        transformation needs to be a class with functions __call__ and __str__

        __call__ will should perform the transformation.
                __call__ receives an image and returns the transformed image


        __str__ should return the name of the path where the images are stored

    """
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    else: 
        return list(pd.read_csv(os.path.join(savedir,saveListOfFiles))["colummn"])

    nbrProcesses = cpu_count() * 2   
    splittedlist = []
    stepsize = len(listOfFiles) // nbrProcesses

    

    #splittedlist = [listOfFiles[i:i + stepsize] for i in range(0, len(listOfFiles), stepsize)]


    for i in range(0,len(listOfFiles),stepsize):
        splittedlist.append(listOfFiles[i:i + stepsize])

            

    jobs = []
    for i in range(len(splittedlist)):
        p = Process(target=target, args= (splittedlist[i],savedir,transformation))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    newListOfFiles = []
    for file in listOfFiles:
        newListOfFiles.append(os.path.join(savedir,file.split("/")[-1]))
    dataframe = pd.DataFrame(newListOfFiles,columns=["colummn"])
    dataframe.to_csv(os.path.join(savedir,saveListOfFiles),index=False)
    return list(pd.read_csv(os.path.join(savedir,saveListOfFiles))["colummn"])





