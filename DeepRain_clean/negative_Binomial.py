#!/home/simon/anaconda3/envs/DeepRain/bin/python
from tensorflow.keras.optimizers import Adam
from Models.Unet import Unet
from Models.Loss import NLL
from Models.Distributions import *
from Models.Utils import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential, Model
from Utils.Dataset import getData
from Utils.transform import cutOut
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
import os

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def param_layer_ZNBinomial(
                        output,
                        parameters=2,
                        dense=256,
                        dropout=0.1,
                        ouput_shape=(64,64)):


    
    layer = output
    layer = MaxPooling2D((2, 2), strides=(2, 2))(layer)
    layer = Conv2D(6, kernel_size=(1, 1), padding="same",activation="selu") (layer)

    cat = Flatten()(layer[:,:,:,:2])
    count = Flatten()(layer[:,:,:,2:4])
    prob = Flatten()(layer[:,:,:,4:6])
    
    cat      = Dense(dense)(cat)
    count    = Dense(dense)(count)
    prob     = Dense(dense)(prob)
    
    
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



def train():
    
    BATCH_SIZE = 20
    DIMENSION = (128,128)
    CHANNELS = 7
    modelpath = "./Models_weights"
    modelname = "ZeroInflatedNegativeBinomial"

    if not os.path.exists(modelpath):
            os.mkdir(modelpath)

    modelpath = os.path.join(modelpath,modelname)

    if not os.path.exists(modelpath):
        os.mkdir(modelpath)


    inputs,outputs = Unet(
                input_shape=(*DIMENSION,CHANNELS),
                output_dim = 6
                )


    y_transform = [cutOut([32,96,32,96])]
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
    plotHistory(history,history_path,title="ZeroInflatedNegativeBinomial NLL-loss")

train()