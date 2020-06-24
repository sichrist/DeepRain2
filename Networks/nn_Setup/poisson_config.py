from tensorflow.keras.optimizers import Adam
from Utils.transform import *
import tensorflow as tf

def NLL(y_true, y_hat):
    return -y_hat.log_prob(y_true)

MODELNAME = "Unet_Poisson"
DIMENSION 					= (128,128)
INPUT_CHANNELS 				= 5
OPTIMIZER 					= Adam( lr= 1e-4 )
BATCHSIZE 					= 100
TIME_TO_PREDICT				= 30
YEARS 						= [2017,2016,2015]

# Cut out the Area around Pxl
slices = [224,352,224,352]
#slices = [32,64+32,32,64+32]
slices_label = [32,64+32,32,64+32]
cutOutFrame = cutOut(slices)
cutOutFrame_label = cutOut(slices_label)

PRETRAINING_TRANSFORMATIONS = [cutOutFrame]
TRANSFORMATIONS 			= [cutOutFrame_label]
LOSSFUNCTION 				= NLL
METRICS 					= ["mse"]

# Custom objects for model loading

CUSTOM_OBJECTS={'exp':tf.exp,'NLL':LOSSFUNCTION}