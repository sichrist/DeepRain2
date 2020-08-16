
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.backend import int_shape
import tensorflow_probability as tfp
tfd = tfp.distributions

def inception_v1(inp,channel,activation="selu"):

    inception_1 = Conv2D(channel, 
                        kernel_size=(1, 1), 
                        padding="same",
                        activation=activation)(inp)
    inception_3 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_5 = Conv2D(channel, 
                        kernel_size=(5, 5), 
                        padding="same",
                        activation=activation)(inp)


    return tf.keras.layers.concatenate([inception_1, 
                                        inception_3, 
                                        inception_5], axis = 3)
    

def inception_v2(inp,channel,activation="selu"):

    inception_1 = Conv2D(channel, 
                        kernel_size=(1, 1), 
                        padding="same",
                        activation=activation)(inp)

    inception_3 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_3_1 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inp)
    inception_3_2 = Conv2D(channel, 
                        kernel_size=(3, 3), 
                        padding="same",
                        activation=activation)(inception_3_1)

    return tf.keras.layers.concatenate([inception_1, 
                                        inception_3, 
                                        inception_3_2], axis = 3)



def lstmLayer(inp,filters = [5,5],activation="selu"):

    shape_inp = int_shape(inp)


    lstm_shape = Reshape((shape_inp[-1],shape_inp[1],shape_inp[2],1))(inp)


    lstm_conv = ConvLSTM2D(filters=filters[0], 
                           kernel_size=(3, 3), 
                           activation=activation,
                           padding='same', 
                           return_sequences=True,
                           data_format='channels_last')(lstm_shape)
    
    

    for i in filters[1:-1]:
        lstm_conv = ConvLSTM2D(filters=i, 
                               kernel_size=(3, 3), 
                               activation=activation,
                               padding='same', 
                               return_sequences=True,
                               data_format='channels_last')(lstm_conv)
        

    lstm_conv = ConvLSTM2D(filters=filters[-1], 
                           kernel_size=(3, 3), 
                           activation=activation,
                           padding='same', 
                           return_sequences=False,
                           data_format='channels_last')(lstm_conv)
    

    return lstm_conv


def CNN_LSTM(input_shape):
    inputs      = Input(shape=input_shape)
    inception_1 = inception_v2(inputs,input_shape[-1])
    lstm_conv1 = lstmLayer(inception_1,filters = [2,5,1])

    inception_2 = inception_v2(inception_1,64)
    inception_3 = inception_v2(inception_2,32)
    inception_4 = inception_v2(inception_2,16)
    #lstm_conv2 = lstmLayer(inception_4,filters = [3,5,1])

    layer = tf.concat([lstm_conv1,inception_4],axis=-1,name="ConcatLayer")
    layer = inception_v2(layer,40)
    layer = SeparableConv2D(3,kernel_size=(3,3))(layer)


    cat = Flatten()(layer[:,:,:,:1])
    count = Flatten()(layer[:,:,:,1:2])
    prob = Flatten()(layer[:,:,:,2:])
    
    cat      = Dense(64)(cat)
    count      = Dense(64)(count)
    prob      = Dense(64)(prob)
    
    
    cat = Dense(64*64,activation="sigmoid")(cat)
    count = Dense(64*64,activation="relu")(count)
    prob = Dense(64*64,activation="sigmoid")(prob)
    
    cat = tf.keras.layers.Reshape((64,64,1))(cat)
    count = tf.keras.layers.Reshape((64,64,1))(count)
    prob = tf.keras.layers.Reshape((64,64,1))(prob)
 
    
    input_dist= tf.concat([cat,count,prob],axis=-1,name="ConcatLayer")

    output_dist = tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(tf.stack([1-tf.math.sigmoid(t[...,:1]), tf.math.sigmoid(t[...,:1])],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
            tfp.distributions.NegativeBinomial(
            total_count=tf.math.softplus(t[..., 1:2]), 
            logits=tf.math.sigmoid(t[..., 2:]) ),])
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 ))

    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)

    return model

def CNN_LSTM(input_shape):
    inputs      = Input(shape=input_shape)
    inception_1 = inception_v2(inputs,input_shape[-1])
    lstm_conv1 = lstmLayer(inception_1,filters = [2,5,1])

    inception_2 = inception_v2(inception_1,64)
    inception_3 = inception_v2(inception_2,32)
    inception_4 = inception_v2(inception_2,16)
    #lstm_conv2 = lstmLayer(inception_4,filters = [3,5,1])

    layer = tf.concat([lstm_conv1,inception_4],axis=-1,name="ConcatLayer")
    layer = inception_v2(layer,40)
    layer = SeparableConv2D(2,kernel_size=(3,3))(layer)


    cat = Flatten()(layer[:,:,:,:1])
    prob = Flatten()(layer[:,:,:,1:])
    
    cat      = Dense(64)(cat)
    prob      = Dense(64)(prob)
    
    
    cat = Dense(64*64,activation="sigmoid")(cat)
    prob = Dense(64*64,activation="sigmoid")(prob)
    
    cat = tf.keras.layers.Reshape((64,64,1))(cat)
    count = tf.keras.layers.Reshape((64,64,1))(count)
    prob = tf.keras.layers.Reshape((64,64,1))(prob)
 
    
    input_dist= tf.concat([cat,prob],axis=-1,name="ConcatLayer")

    def ZeroInflated_Poisson():
    return tfp.layers.DistributionLambda(
          name="DistributionLayer",
          make_distribution_fn=lambda t: tfp.distributions.Independent(
          tfd.Mixture(
              cat=tfd.Categorical(probs=tf.stack([1-t[...,0:1], t[...,0:1]],axis=-1)),
              components=[tfd.Deterministic(loc=tf.zeros_like(t[...,0:1])),
              tfd.Poisson(rate=tf.math.softplus(t[...,1:2]))]),
          name="ZeroInflated",reinterpreted_batch_ndims=0 ))
    output = output_dist(input_dist)
    model = Model(inputs=inputs, outputs=output)

    return model