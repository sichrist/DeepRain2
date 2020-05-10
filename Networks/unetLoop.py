from Utils.loadset import getDataSet
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
import os
from trainer import Trainer
try:
    from Utils.connection_cfg import *
except Exception as e:
    PSWD = None
    USRN = None
    
from Utils.Data import dataWrapper
from Utils.transform import ToCategorical, cutOut

def convLayer(inp,nbrLayer,channel,activation="selu"):
    assert nbrLayer > 0, "In Function convLayer nbrLayer > 0 ?"
    layer = Conv2D(channel, kernel_size=(3, 3), padding="same") (inp)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    for i in range(1,nbrLayer):
        layer = Conv2D(channel, kernel_size=(3, 3), padding="same")  (layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization()(layer)
    return layer

#def FullUnetLoop(input_shape,down_channels=[64,128,256,512,1024],downLayer=2,activation="relu"):

def FullUnetLoopZero(input_shape,down_channels=[64,128,256,512],downLayer=2,activation="relu"):

    def zeroInflatedPoisson(output):
        rate = tf.math.exp(output[:,:,:,0:1]) #A 
        s = tf.math.sigmoid(output[:,:,:,1:2])
        
        pos = tfp.distributions.Poisson(rate=rate)
        det = tfp.distributions.Deterministic(loc=tf.zeros_like(rate))
        components = [det,pos]
        mixture = tfd.Mixture(
              cat=tfd.Categorical(probs=tf.stack([1-s, s],axis=-1)),#D
              components=components)
        return tfp.distributions.Independent(mixture,reinterpreted_batch_ndims=1,name="ZeroInflated")
        #return mixture
    
    inputs = Input(shape=input_shape)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same") (inputs)
    layer = Activation(activation)(layer)
    layer = BatchNormalization()(layer)
    
    layer = Conv2D(down_channels[0], kernel_size=(3, 3), padding="same") (layer)
    layer = Activation(activation)(layer)
    firstLayer = BatchNormalization()(layer)
    
    pool  = MaxPooling2D((2, 2), strides=(2, 2))(firstLayer)
    
    layerArray = []
    
    for channel in down_channels[1:]:
        
        layer = convLayer(pool,downLayer,channel)
       
        if channel != down_channels[-1]:
            layerArray.append(layer)
            pool  = MaxPooling2D((2, 2), strides=(2, 2))(layer)
            
    for i,channel in enumerate(reversed(down_channels[:-1])):
        
        layer = Conv2DTranspose(channel,(3, 3),strides=(2,2),padding="same")(layer)
        layer = Activation(activation)(layer)
        layer = BatchNormalization() (layer)
        
        if len(layerArray) >= (i+1):
            layer = concatenate([layerArray[-(i+1)], layer], axis=3)
        else:
            layer = concatenate([firstLayer, layer], axis=3)
        
        layer = convLayer(layer,downLayer,channel)
        
    output = Conv2D(1, kernel_size=(1, 1), padding="same") (layer)

    output = Flatten()(output)
    
    #output = tfp.layers.DistributionLambda(zeroInflatedPoisson,name="ZPoisson")(output)
    output = tfp.layers.IndependentPoisson((64,64,1))(output)
    
    model = Model(inputs=inputs, outputs=output)
    return model

dimension = (64,64)
batch_size = 100
channels = 5
optimizer = Adam( lr = 1e-3 )
slices = [256,320,256,320]
cutOutFrame = cutOut(slices)

categorical_list = [0,1,5,10,15,30,60,120]
categorical = ToCategorical(categorical_list)

PRETRAINING_TRANSFORMATIONS = [cutOutFrame]
TRANSFORMATIONS = [categorical]
TRANSFORMATIONS = None

def NLL(y_true, y_hat):
    return -y_hat.log_prob(y_true)


def provideData(flatten=False,dimension=dimension,batch_size=60,transform=None,preTransformation=None):

    getDataSet(DatasetFolder,year=[2017],username=USRN,pswd=PSWD)
    train,test = dataWrapper(PathToData,
                            dimension=dimension,
                            channels=channels,
                            batch_size=batch_size,
                            overwritecsv=True,
                            flatten=flatten,
                            onlyUseYears=[2017],
                            transform=transform,
                            preTransformation=preTransformation)
    
    return train,test
DatasetFolder = "./Data/RAW"
PathToData = os.path.join(DatasetFolder,"MonthPNGData")

train, test = provideData(dimension=dimension,
                          batch_size=batch_size,
                          transform=TRANSFORMATIONS,
                          preTransformation=PRETRAINING_TRANSFORMATIONS)

t = Trainer(FullUnetLoopZero,
                    NLL,
                    (train,test),
                    batch_size = batch_size,
                    optimizer=optimizer,
                    dimension = dimension,
                    channels = channels,
                    metrics = ["mse","mae"])

t.fit(80)
