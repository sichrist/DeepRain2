#!/home/simon/anaconda3/envs/DeepRain/bin/python
from nn_Setup import poisson_config
from Models.Unet import poisson_unet
from trainer import Trainer
from Utils.Data import provideData


train, test = provideData(dimension  		= poisson_config.DIMENSION,
                          channels   		= poisson_config.INPUT_CHANNELS,
                          timeToPred 		= poisson_config.TIME_TO_PREDICT,
                          batch_size 		= poisson_config.BATCHSIZE,
                          transform  		= poisson_config.TRANSFORMATIONS,
                          year              = poisson_config.YEARS,
                          preTransformation = poisson_config.PRETRAINING_TRANSFORMATIONS)

model = Trainer(poisson_unet,
                    poisson_config.NLL,
                    (train,test),
                    batch_size = poisson_config.BATCHSIZE,
                    optimizer  = poisson_config.OPTIMIZER,
                    dimension  = poisson_config.DIMENSION,
                    channels   = poisson_config.INPUT_CHANNELS,
                    metrics    = poisson_config.METRICS,
                    modelname  = poisson_config.MODELNAME)


model.fit(30)