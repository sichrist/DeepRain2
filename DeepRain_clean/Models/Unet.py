
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def convLayer(inp,
        nbrLayer,
        channel,
        activation="relu",
        bias_regularizer = 0.01,
        kernel_regularizer = 0.01):

    assert nbrLayer > 0, "In Function convLayer nbrLayer > 0 ?"
    layer = Conv2D(channel, kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (inp)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    for i in range(1,nbrLayer):
        layer = Conv2D(channel, kernel_size=(3, 3), padding="same",
            kernel_regularizer=l2(kernel_regularizer), 
            bias_regularizer=l2(bias_regularizer))  (layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization()(layer)
    return layer

def Unet(input_shape,
        down_channels=[64,128,256,512],
        downLayer=2,
        activation="selu",
        output_activation = "selu",
        output_dim = 1,
        bias_regularizer = 0.01,
        kernel_regularizer = 0.01,
        ):
    
    inputs = Input(shape=input_shape)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (inputs)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same",
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (layer)
    layer = Activation(activation)(layer)
    firstLayer = BatchNormalization()(layer)
    
    pool  = MaxPooling2D((2, 2), strides=(2, 2))(firstLayer)
    
    layerArray = []
    
    for channel in down_channels[1:]:
        
        layer = convLayer(pool,
                        downLayer,
                        channel,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)
       
        if channel != down_channels[-1]:
            layerArray.append(layer)
            pool  = MaxPooling2D((2, 2), strides=(2, 2))(layer)
            
    for i,channel in enumerate(reversed(down_channels[:-1])):
        
        layer = Conv2DTranspose(channel,(3, 3),strides=(2,2),padding="same",
            kernel_regularizer=l2(kernel_regularizer), 
            bias_regularizer=l2(bias_regularizer))(layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization() (layer)
        
        if len(layerArray) >= (i+1):
            layer = concatenate([layerArray[-(i+1)], layer], axis=3)
        else:
            layer = concatenate([firstLayer, layer], axis=3)
        
        layer = convLayer(layer,
            downLayer,
            channel,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
            )
        
    output = Conv2D(output_dim, kernel_size=(1, 1), padding="same",
        activation=output_activation,
        kernel_regularizer=l2(kernel_regularizer), 
        bias_regularizer=l2(bias_regularizer)) (layer)
    return inputs,output
    #return Model(inputs=inputs, outputs=output)
    