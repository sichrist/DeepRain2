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

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)