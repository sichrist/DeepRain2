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
import os

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

    layer_1 = Flatten()(output[:,:,:,:1])
    layer_2 = Flatten()(output[:,:,:,1:2])
    
    layer_1      = Dense(dense)(layer_1)
    layer_2      = Dense(dense)(layer_2)
    layer_1      = Dropout(dropout)(layer_1)
    layer_1      = Dropout(dropout)(layer_1)
    
    layer_1 = Dense(ouput_shape[0]*ouput_shape[1],activation="sigmoid")(layer_1)
    layer_2 = Dense(ouput_shape[0]*ouput_shape[1],activation="relu")(layer_2)
    layer_1 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_1)
    layer_2 = tf.keras.layers.Reshape((*ouput_shape,1))(layer_2)
    input_dist= tf.concat([layer_1,layer_2],axis=-1)

    return input_dist


def train_Unet_zeroPoisson():

    BATCH_SIZE = 10
    DIMENSION = (128,128)
    CHANNELS = 7
    modelpath = "./Models_weights"
    modelname = "ZeroInflatedPoisson"




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
    modelpath = os.path.join(modelpath,
                            modelname+'-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5')

    checkpoint = ModelCheckpoint(modelpath,
                                 verbose=0,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='auto')



    history = model.fit(train,
                        validation_data = test,
                        shuffle         = True,
                        epochs          = 10,
                        batch_size      = BATCH_SIZE,
                        callbacks       = checkpoint)




train_Unet_zeroPoisson()

