from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.data_utils import Sequence
from Utils.loadset import getDataSet
from .transform import transformImages,splitImgProcess, ImageToPatches
import os
CSVFILE = "./.listOfFiles.csv"
WRKDIR = "./Data"
TRAINSETFOLDER=os.path.join(WRKDIR,"train")
VALSETFOLDER=os.path.join(WRKDIR,"val")

DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder,"MonthPNGData")

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

def prepareListOfFiles(path,
        workingdir = WRKDIR,
        nameOfCsvFile=CSVFILE,
        sortOut=False,
        overwritecsv=False,
        onlyUseYears=None):
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

class Dataset(Sequence):
    """docstring for Dataset"""
    def __init__(self, 
                year=[2017],
                timeToPred = 30,
                outputshape = (256,256),
                inputshape = (256,256),
                workingdir = WRKDIR,
                batchsize = 1
                ):
        super(Dataset, self).__init__()
        self.year       = year
        self.timeToPred = timeToPred
        self.timeSteps  = 5
        self.batchsize  = batchsize
        self.cutTheFuckOut = 50

        listofFiles = prepareListOfFiles(PathToData,onlyUseYears=year)
        savedir = str(inputshape[0])+"x"+str(inputshape[1])+"_"+str(outputshape[0])+"x"+str(outputshape[1])
        newlistcsv = savedir+".csv"

        idcs = np.arange(len(listofFiles))
        sortedOutIdcs = []

        img = Image.open(listofFiles[0])
        patches = ImageToPatches(outputsize=outputshape,inputsize=inputshape)
        x,y = patches(np.array(img))
        
        #self.length = nPerImg = x.shape[0] * x.shape[1]

    def cutEdges(self,img):
        return img[self.cutTheFuckOut:-self.cutTheFuckOut,self.cutTheFuckOut:-self.cutTheFuckOut,:]


    def __len__(self):
        pass

    def __call__(self,index):

            



