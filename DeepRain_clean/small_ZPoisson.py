#!/home/simon/anaconda3/envs/DeepRain/bin/python
from tensorflow.keras.optimizers import Adam
from Models.Unet import Unet
from Models.Loss import NLL, KLL
from Models.Distributions import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model
from Utils.Dataset import getData
from Utils.transform import cutOut
from tensorflow.keras.callbacks import *
from Models.Utils import *
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import os
import cv2 as cv
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
                        ouput_shape=(12,12),
                        kernel_regularizer=l2(0.01), 
                        bias_regularizer=l2(0.01)):

    layer = output
    #layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = Conv2D(128, 
                    kernel_size=(5, 5), 
                    padding="same",
                    activation="selu",
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer) (layer)

    layer_1 = Flatten()(layer[:,:,:,:1])
    layer_2 = Flatten()(layer[:,:,:,1:2])
    
    layer_1      = Dense(dense,
                    bias_initializer="glorot_uniform",
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer)(layer_1)
    layer_2      = Dense(dense,
                    bias_initializer="glorot_uniform",
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer)(layer_2)

    #layer_1      = Dropout(dropout)(layer_1)
    #layer_2      = Dropout(dropout)(layer_2)
    
    layer_1 = Dense(ouput_shape[0]*ouput_shape[1],
                    activation="sigmoid",
                    bias_initializer="glorot_uniform")(layer_1)
    layer_2 = Dense(ouput_shape[0]*ouput_shape[1],
                    activation="relu",
                    bias_initializer="glorot_uniform")(layer_2)
    layer_1 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_1)
    layer_2 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_2)
    input_dist= tf.concat([layer_1,layer_2],axis=-1)

    return input_dist


BATCH_SIZE = 100
DIMENSION = (16,16)
CHANNELS = 5
MODELPATH = "./Models_weights"
MODELNAME = "small_ZeroInflatedPoisson"


def getModel():

    modelpath = MODELPATH
    modelname = MODELNAME
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)




    inputs,outputs = Unet(
                down_channels=[64,96,128,192],
                input_shape=(*DIMENSION,CHANNELS),
                output_dim = 144,
                kernel_regularizer = 0.01,
                bias_regularizer = 0.01
                )

    y_transform = [cutOut([2,14,2,14])]
    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         timeToPred=5,
                         y_transform=y_transform)


    outputs = param_layer_ZPoisson(outputs,
                kernel_regularizer = l2(0.0),
                bias_regularizer = l2(0.0))
    dist_outputs = ZeroInflated_Poisson()
    outputs = dist_outputs(outputs)


    neg_log_likelihood = lambda x, rv_x: tf.math.reduce_mean(-rv_x.log_prob(x))
    model = Model(inputs,outputs)
    model.compile(loss=neg_log_likelihood,
                  optimizer=Adam( lr= 1e-3 ))
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
                            epochs          = 20+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 100,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="small ZeroInflatedPoisson NLL-loss")



def eval():
    #windowname = 'OpenCvFrame'
    #cv.namedWindow(windowname)
    #cv.moveWindow(windowname,0,00)
    modelpath = MODELPATH
    modelname = MODELNAME

    values = 256
    ones = np.ones((1,12,12,1),np.float32)
    value_array = np.repeat(ones,values,axis=-1)

    for i in range(values):
        value_array[:,:,:,i] *= i


    def probs(distribution):
        prob_array = np.zeros_like(value_array)
        for i in range(values):
            prob_array[:,:,:,i:i+1] = distribution.prob(value_array[:,:,:,i:i+1])
        print(prob_array.argmax(-1).max())
        return prob_array

    model,checkpoint,modelpath,train,test = getModel()   
    predictions = []
    for nbr,(x,y) in enumerate(test):
        print("{:7d}|{:7d}".format(nbr,len(test)),end="\r")

        for i in range(BATCH_SIZE):
            pred = model(np.array([x[i,:,:,:]]))
            p = np.array(pred.mean())
 
            print(y[i,:,:].max(),np.max(pred.prob(y[i,:,:].max())) )
            
            
    np.save(os.path.join(modelpath,modelname+"_predictions"),np.array(predictions))



#train()
eval()




