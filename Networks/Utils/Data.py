 
from PIL import Image
import numpy as np
import pandas as pd
import re
import keras
import os
from .transform import transformImages,resize
import cv2
from Utils.colors import *
from tensorflow.python.keras.utils.data_utils import Sequence
from Utils.loadset import getDataSet


CSVFILE = "./.listOfFiles.csv"
WRKDIR = "./Data"
TRAINSETFOLDER=os.path.join(WRKDIR,"train")
VALSETFOLDER=os.path.join(WRKDIR,"val")

DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder,"MonthPNGData")


def provideData(dimension,
                batch_size,
                channels,
                transform=None,
                preTransformation=None,
                year = [2017],
                onlyUseYears = None,
                DatasetFolder=DatasetFolder):

    """
        DatasetFolder       : Download Data to this Folder an use it as working directory
        preTransformation   : Transformation to perform on Data BEFORE loading it
        transform           : Transformation to perform on a single Image before hand it to the NN
        onlyUseYears        : List of years to use in Dataset

    """

    getDataSet(DatasetFolder,year=[2017])
    train,test = dataWrapper(PathToData,
                            dimension=dimension,
                            channels=channels,
                            batch_size=batch_size,
                            overwritecsv=True,
                            onlyUseYears=onlyUseYears,
                            transform=transform,
                            preTransformation=preTransformation)
    
    return train,test

def dataWrapper(path,
                dimension,
                channels,
                batch_size,
                csvFile=CSVFILE,
                workingdir=WRKDIR,
                split=0.25,
                flatten=False,
                sortOut=True,
                shuffle=True,
                overwritecsv=False,
                onlyUseYears=None,
                preTransformation=None,
                transform=None):
    """
    
        Returns two Data objects, which can be used for training.
        The first returned object is the trainingdata, second is
        testdata.

        path        : path to Data
        channels    : number of channels used as input
        batch_size  : self-explanatory
        csvFile     : name of csvFile where path to files are stored
                      (the benefit is that it is not necessary to load
                      the whole set into RAM, which could be too large)
        workingdir  : Directory where Data is expanded (resized image are
                      stored)
        split       : split ratio (train/test)
        sortOut     : not used
        shuffle     : shuffle Data after epochs,
        onlyUseYears: [2016,2017] ... make sure to use years in list

    """

    
    data = prepareListOfFiles(path,sortOut=sortOut,overwritecsv=overwritecsv,onlyUseYears=onlyUseYears)
    trainingsSet,validationSet = splitData(data)

    if not os.path.exists(TRAINSETFOLDER):
        os.mkdir(TRAINSETFOLDER)
    if not os.path.exists(VALSETFOLDER):
        os.mkdir(VALSETFOLDER)


    filename,ext = os.path.splitext(csvFile) 
    trainsetCSV = filename+"_train_"+ext
    valsetCSV = filename+"_val_"+ext


    train_dataframe = pd.DataFrame(trainingsSet,columns=["colummn"])
    train_dataframe.to_csv(os.path.join(TRAINSETFOLDER,trainsetCSV),index=False)
    val_dataframe = pd.DataFrame(validationSet,columns=["colummn"])
    val_dataframe.to_csv(os.path.join(VALSETFOLDER,valsetCSV),index=False)

    train = Dataset(TRAINSETFOLDER ,
                    dim = dimension,
                    n_channels = channels,
                    batch_size = batch_size,
                    workingdir=TRAINSETFOLDER,
                    saveListOfFiles=trainsetCSV,
                    flatten=flatten,
                    shuffle=shuffle,
                    transform=transform,
                    preTransformation=preTransformation)

    val = Dataset(VALSETFOLDER,
                    dim = dimension,
                    n_channels = channels,
                    batch_size = batch_size,
                    workingdir=VALSETFOLDER,
                    saveListOfFiles=valsetCSV,
                    flatten=flatten,
                    shuffle=shuffle,
                    transform=transform,
                    preTransformation=preTransformation)


    return train,val


def splitData(data,split=0.25):
    
    dataLength = len(data)
    validation_length = int(np.floor(dataLength * split))

    validationSet = data[-validation_length:]
    trainingsSet = data[:-validation_length]


    return trainingsSet,validationSet


def prepareListOfFiles(path,workingdir = WRKDIR,nameOfCsvFile=CSVFILE,sortOut=False,overwritecsv=False,onlyUseYears=None):
    if not os.path.exists(workingdir):
        os.mkdir(workingdir)

    if not os.path.exists(os.path.join(workingdir,nameOfCsvFile)) or overwritecsv or onlyUseYears is not None:

        listOfFiles = getListOfFiles(path)
        listOfFiles.sort()

        if onlyUseYears:
            newlist = []
            prefix = "YW2017.002_"
            for year in onlyUseYears:
                checkFile = prefix + str(year)
                for file in listOfFiles:
                    if checkFile in file:
                        newlist.append(file)
            listOfFiles = newlist



        dataframe = pd.DataFrame(listOfFiles,columns=["colummn"])
        dataframe.to_csv(os.path.join(workingdir,nameOfCsvFile),index=False)
        listOfFiles = dataframe
    
    listOfFiles = list(pd.read_csv(os.path.join(workingdir,nameOfCsvFile))["colummn"])

    return listOfFiles


def getListOfFiles(path):
    """
    
        stolen from :
        https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/

    """


    directory_entries = os.listdir(path)
    files = []

    for entry in directory_entries:
        fullPath = os.path.join(path,entry)
        if os.path.isdir(fullPath):
            files = files + getListOfFiles(fullPath)
        else:
            files.append(fullPath)
    return files


def mergeLists(list1,list2):
    newlist = []
    for file in list1:
        name = file.split("/")[-1]
        for file2 in list2:
            name2 = file2.split("/")[-1]

            if name == name2:
                newlist.append(file)
    return newlist

class Dataset(Sequence):

    def __init__(self,path,
                      batch_size,
                      dim,
                      n_channels=5,
                      shuffle=True,
                      saveListOfFiles=CSVFILE,
                      workingdir=WRKDIR,
                      timeToPred = 5,
                      timeSteps = 5,
                      sequenceExist = False,
                      flatten = False,
                      sortOut=True,
                      lstm=True,
                      dtype=np.float32,
                      transform=None,
                      preTransformation=None):


        """
        
            timeToPred    : minutes forecast, default is 30 minutes
            timeSteps     : time between images, default is 5 minutes

        """
        assert batch_size > 0, "batch_size needs to be greater than 0"
        assert timeSteps % 5 == 0, "timesteps % 5 needs to be 0"
        assert timeToPred % 5 == 0, "timeToPred % 5 needs to be 0"


        self.path = path
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.workingdir = workingdir
        self.saveListOfFiles = saveListOfFiles
        self.timeToPred = timeToPred
        self.timeSteps = timeSteps
        self.steps = int(timeToPred / timeSteps)
        self.datatype = dtype
        self.flatten = flatten
        self.sortOut = sortOut
        self.lstm = lstm
        self.transform = transform
        self.preTransformation = preTransformation


        if preTransformation is None:
            self.preTransformation=[resize(self.dim)]
        else:
            self.preTransformation=preTransformation

        # index offset
        self.label_offset = self.n_channels + self.steps - 1

        if not os.path.exists(self.workingdir):
            os.mkdir(self.workingdir)

        if not os.path.exists(os.path.join(self.workingdir,saveListOfFiles)):
            self.listOfFiles = getListOfFiles(self.path)
            self.listOfFiles.sort()
            dataframe = pd.DataFrame(self.listOfFiles,columns=["colummn"])
            dataframe.to_csv(os.path.join(self.workingdir,saveListOfFiles),index=False)
            self.listOfFiles = dataframe

        if type(self.preTransformation) is list:
            pathName = ""
            for i in range(0,len(self.preTransformation)-1):
                transObj = self.preTransformation[i]
                pathName += str(transObj)
            pathName += str(self.preTransformation[-1])
            savefolder = pathName

        else:
            savefolder = str(self.preTransformation)

        self.listOfFiles = list(pd.read_csv(os.path.join(self.workingdir,saveListOfFiles))["colummn"])


     
        self.new_listOfFiles = transformImages(self.listOfFiles,self.preTransformation,os.path.join(workingdir,savefolder),saveListOfFiles)

        if len(self.new_listOfFiles) != len(self.listOfFiles):
            print(YELLOW + "WARNING: Length of lists does not match! " + RESET)
            print(YELLOW +"To stop this warning, delete {} folder and restart".format(os.path.join(workingdir,savefolder))+RESET)
            print(YELLOW +"Trying to update..\n"+ RESET)

            newlist = mergeLists(self.new_listOfFiles,self.listOfFiles)
            if len(newlist) == 0:
                print(RED+"Could not fix this Problem!!! abort"+RESET)
                exit(-1)

            print(GREEN+"Fixed! "+RESET)
            self.new_listOfFiles = newlist
            if len(self.new_listOfFiles) != len(self.listOfFiles):
                print(CYAN+"But length does not match again.... just delete the folders bruh"+RESET)

        self.listOfFiles = self.new_listOfFiles
        #self.listOfFiles = self.new_listOfFiles[416:436]
        #self.listOfFiles = self.new_listOfFiles[1000:1500]

        #self.indizes = np.arange(len(self))
        self.indizes = np.arange(len(self.listOfFiles)-self.label_offset)

        def sortOut(listOfFiles):
            y = []
            for i in range(len(listOfFiles) -self.label_offset):
                path = listOfFiles[i]
                img = np.array(Image.open(path))
                if img.max() > 0:
                    y.append(i)
            return y
        if self.sortOut:
            self.indizes = sortOut(self.listOfFiles)

        


    def __data_generation(self,index):
        X = []
        Y = []
        
        for i,id in enumerate(range(index,index+self.n_channels)):
            
            img = np.array(Image.open(self.listOfFiles[id]))
            
            assert img.shape == self.dim, \
            "[Error] (Data generation) Image shape {} does not match dimension {}".format(img.shape,self.dim)            

            X.append(img)


        try:
            label = np.array(Image.open(self.listOfFiles[index+self.label_offset]))
            
        except Exception as e:
            print("\n\n",index,self.label_offset,len(self.listOfFiles))
            exit(-1)



        if self.transform is not None:
            for operation in self.transform:
                label = operation(label)

                    
        
        
        #Y.append(label.flatten())
        #Y = label.flatten()
        #Y = label
        Y.append(label)
        

        return np.array(X),np.array(Y)


    def on_epoch_end(self):
        
        if self.shuffle == True:
            np.random.shuffle(self.indizes)



    def __len__(self):
        #return int(np.floor((len(self.listOfFiles) - self.label_offset)/self.batch_size )) 
        return int(np.floor((len(self.indizes) - self.label_offset)/self.batch_size ))

 
    def __getitem__(self,index):
        
        if index >= len(self)-1:
            self.on_epoch_end()

        X = []
        Y = []
        
        id_list = self.indizes[(index*self.batch_size):(index*self.batch_size)+self.batch_size]
        
        for idd in id_list:
            
            x,y = self.__data_generation(idd)

            X.append(x)
            Y.append(y)
        
        X = np.array(X)
        Y = np.array(Y)
        
        
        #print(X.shape,Y.shape)

        X = np.transpose(X,(0,2,3,1))
        if self.transform is not None:
            Y = np.transpose(Y,(0,2,3,1))
            
            return X/255.0,Y/255.0
        if not self.flatten:
            pass
            #Y = np.transpose(Y,(0,2,3,1))
        else:
           return X/255,Y.flatten()



        return X/255.0,Y/255.0
        
