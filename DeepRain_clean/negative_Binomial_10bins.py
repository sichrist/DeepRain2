#!/home/simon/anaconda3/envs/DeepRain/bin/python
from tensorflow.keras.optimizers import Adam
from Models.Unet import Unet
from Models.Loss import NLL
from Models.Distributions import *
from Models.Utils import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model
from Utils.Dataset import getData
from Utils.transform import cutOut,LinBin
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import os
import cv2 as cv

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def param_layer_ZNBinomial(
                        output,
                        parameters=2,
                        dense=256,
                        dropout=0.1,
                        ouput_shape=(64,64),
                        kernel_regularizer=l2(0.01), 
                        bias_regularizer=l2(0.01)):


    
    layer = output
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = Conv2D(6, kernel_size=(1, 1), padding="same",activation="selu") (layer)

    cat = Flatten()(layer[:,:,:,:2])
    count = Flatten()(layer[:,:,:,2:4])
    prob = Flatten()(layer[:,:,:,4:6])
    
    cat      = Dense(dense,
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer)(cat)
    count    = Dense(dense,
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer)(count)
    prob     = Dense(dense,
                    kernel_regularizer=kernel_regularizer, 
                    bias_regularizer=bias_regularizer)(prob)
    
    
    #cat     = Dropout(dropout)(cat)
    #count   = Dropout(dropout)(count)
    #prob    = Dropout(dropout)(prob)
    
    cat     = Dense(ouput_shape[0]*ouput_shape[1],activation="sigmoid")(cat)
    count   = Dense(ouput_shape[0]*ouput_shape[1],activation="relu")(count)
    prob    = Dense(ouput_shape[0]*ouput_shape[1],activation="sigmoid")(prob)
    
    cat     = tf.keras.layers.Reshape((*ouput_shape,1))(cat)
    count   = tf.keras.layers.Reshape((*ouput_shape,1))(count)
    prob    = tf.keras.layers.Reshape((*ouput_shape,1))(prob)
    
    input_dist = tf.concat([cat,count,prob],axis=-1)


    return input_dist

BATCH_SIZE = 20
DIMENSION = (128,128)
CHANNELS = 7
MODELPATH = "./Models_weights"
MODELNAME = "ZeroInflatedNegativeBinomial_10bins"

def getModel():
    modelpath = MODELPATH
    modelname = MODELNAME

    binit = LinBin()
    if not os.path.exists(modelpath):
            os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)


    inputs,outputs = Unet(
                input_shape=(*DIMENSION,CHANNELS),
                output_dim = 6
                )


    y_transform = [cutOut([32,96,32,96]),binit]
    
    train,test = getData(BATCH_SIZE,
                         DIMENSION,CHANNELS,
                         y_transform=y_transform)


    outputs = param_layer_ZNBinomial(outputs)
    dist_outputs = ZeroInflated_Binomial()
    outputs = dist_outputs(outputs)

    model = Model(inputs,outputs)
    model.compile(loss=NLL,
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
                            epochs          = 10+epoch,
                            initial_epoch   = epoch,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = mergeHist(laststate["history"],history.history)

    else:
        history = model.fit(train,
                            validation_data = test,
                            shuffle         = True,
                            epochs          = 20,
                            batch_size      = BATCH_SIZE,
                            callbacks       = checkpoint)

        history = history.history



    saveHistory(history_path,history)
    plotHistory(history,history_path,title="ZeroInflatedNegativeBinomial_10bins NLL-loss")


def eval():
    windowname = 'OpenCvFrame'
    cv.namedWindow(windowname)
    cv.moveWindow(windowname,0,00)
    modelpath = MODELPATH
    modelname = MODELNAME

    values = 256
    ones = np.ones((1,64,64,1),dtype=np.float32)
    value_array = np.repeat(ones,values,axis=-1)
    for i in range(values):
        value_array[:,:,:,i] *= i/values


    value_array = tf.convert_to_tensor(value_array,dtype=tf.float32)
    print(value_array)
    def probs(distribution):
        prob_array = np.zeros_like(value_array)
        for i in range(values):
            prob_array[:,:,:,i:i+1] = distribution.prob(value_array[:,:,:,i:i+1])
            #prob_array[:,:,:,i:i+1] = distribution.sample()
            #print(distribution.mean())

        #print(prob_array[0,0,:])
        print(prob_array.argmax(-1).max())
        return prob_array

    model,checkpoint,modelpath,train,test = getModel()   
    predictions = []
    for nbr,(x,y) in enumerate(test):
        print("{:7d}|{:7d}".format(nbr,len(test)),end="\r")
        
        for i in range(BATCH_SIZE):
            pred = model(np.array([x[i,:,:,:]]))
            predictions.append((np.array(y[i,:,:]),probs(pred)))
            p = np.array(pred.mean())
            #print(p)
            cv.imshow(windowname,p[0,:,:,:])
            if cv.waitKey(25) & 0XFF == ord('q'):
                    break

    np.save(os.path.join(modelpath,modelname+"_predictions"),np.array(predictions))

#train()

eval()