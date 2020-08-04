#!/home/simon/anaconda3/envs/DeepRain/bin/python
from tensorflow.keras.optimizers import Adam
from Models.Unet import Unet
from Models.Loss import NLL
from Models.Distributions import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model
from Utils.Dataset import getData
from Utils.transform import cutOut
from tensorflow.keras.callbacks import *
from Models.Utils import *
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

def param_layer_ZPoisson(
                        output,
                        parameters=2,
                        dense=256,
                        dropout=0.1,
                        ouput_shape=(64,64)):

    layer = output
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = Conv2D(2, kernel_size=(1, 1), padding="same",activation="selu") (layer)

    layer_1 = Flatten()(layer[:,:,:,:1])
    layer_2 = Flatten()(layer[:,:,:,1:2])
    
    layer_1      = Dense(dense)(layer_1)
    layer_2      = Dense(dense)(layer_2)
    layer_1      = Dropout(dropout)(layer_1)
    layer_1      = Dropout(dropout)(layer_1)
    
    layer_1 = Dense(ouput_shape[0]*ouput_shape[1],activation="sigmoid")(layer_1)
    layer_2 = Dense(ouput_shape[0]*ouput_shape[1],activation="selu")(layer_2)
    layer_1 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_1)
    layer_2 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_2)
    input_dist= tf.concat([layer_1,layer_2],axis=-1)

    return input_dist


BATCH_SIZE = 20
DIMENSION = (128,128)
CHANNELS = 7
MODELPATH = "./Models_weights"
MODELNAME = "ZeroInflatedPoisson"

def getModel():

    modelpath = MODELPATH
    modelname = MODELNAME
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)




    inputs,outputs = Unet(
                input_shape=(*DIMENSION,CHANNELS),
                output_dim = 2
                )

    y_transform = [cutOut([32,96,32,96])]
    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         y_transform=y_transform)


    outputs = param_layer_ZPoisson(outputs)
    dist_outputs = ZeroInflated_Poisson()
    outputs = dist_outputs(outputs)


    model = Model(inputs,outputs)
    model.compile(loss=NLL,
                  optimizer=Adam( lr= 1e-4 ))
    model.summary()
    modelpath_h5 = os.path.join(modelpath,
                            modelname+'-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5')

    checkpoint = ModelCheckpoint(modelpath_h5,
                                 verbose=0,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto')

    return model,checkpoint,modelpath,train,test

def train():
    modelpath = MODELPATH
    modelname = MODELNAME

    model,checkpoint,modelpath,train,test = getModel()

    history_path = os.path.join(modelpath,modelname+"_history")
    laststate = getBestState(modelpath,history_path)
    test.setWiggle_off()

    
    if laststate:
        epoch = laststate["epoch"]
        model.load_weights(laststate["modelpath"])

        loss = model.evaluate(x=test, verbose=2)
        print("Restored model, loss: {:5.5f}".format(loss))

        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 10+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 10,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="ZeroInflatedPoisson NLL-loss")



def eval():
    #windowname = 'OpenCvFrame'
    #cv2.namedWindow(windowname)
    #cv2.moveWindow(windowname,750,1280)

    values = 256
    ones = np.ones((1,64,64,1),dtype=np.uint8)
    value_array = np.repeat(ones,values,axis=-1)
    for i in range(values):
        value_array[:,:,:,i] *= i

    def interval(predictions):
        import seaborn as sns
        import matplotlib.pyplot as plt





    def quantile(distribution):
        prob_array = np.zeros_like(value_array)
        for i in range(values):
            prob_array[:,:,:,i:i+1] = distribution.cdf(value_array[:,:,:,i:i+1])

        return prob_array
        #p = prob_array.argmax(axis=-1)
        #print(p,p.shape)

        #for i in range(values):
        #    print(prob_array[0,0,i],end=" ")
        #exit(-1)
        #print()



    modelpath = MODELPATH
    modelname = MODELNAME

    model,checkpoint,modelpath,train,test = getModel()   
    predictions = []
    for nbr,(x,y) in enumerate(test):
        print("{:7d}|{:7d}".format(nbr,len(test)))
        
        
        for i in range(BATCH_SIZE):
            pred = model(np.array([x[i,:,:,:]]))
            predictions.append((pred,np.array(y[i:,:,:]),quantile(pred)))

        if nbr == 10:
            break



train()
#eval()




