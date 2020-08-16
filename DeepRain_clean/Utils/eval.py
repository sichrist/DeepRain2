from multiprocessing import Process, cpu_count, Manager, Queue
import numpy as np
from time import sleep
import psutil
import tensorflow as tf
import tensorflow_probability as tfp
from keras.backend import clear_session
import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
tfd = tfp.distributions



def sleepIfRAMFULL(threshold=75,time_sleep=5):
    ram_percent = psutil.virtual_memory().percent
    if (ram_percent > threshold):
        print("MainThread RAM-usage: {:.2f} .. going to sleep for {}s".format(ram_percent,time_sleep),end="\r")
        sleep(time_sleep)
    

def dist2Classes(p,threshold = 0.5):
    # Wahrscheinlichkeit für kein Regen
    
    rain = 1 - np.array(p.prob(0))
    if np.isnan(rain).any():
        print(np.isnan(rain).sum())
    
    
    # Wahrscheinlichkeit Regen größer threshold, 
    mask = (rain > threshold)
    

    return mask

def labelToMask(label):
    # Gibt Maske mit Regen zurück
    return (label > 0)


def get_TP(simple,y,pred,threshold=0.5):

    
    true_pos = np.zeros(y.shape)
    true_neg = true_pos.copy()
    false_pos = true_pos.copy()
    false_neg = true_pos.copy()
    rain_total = 0
    total = 0

    simple_true_pos =  true_pos.copy()
    simple_true_neg =  true_pos.copy()
    simple_false_pos = true_pos.copy()
    simple_false_neg = true_pos.copy()

            
    pred_mask = dist2Classes(pred,threshold=threshold)

    labl_mask = labelToMask(y)

    #              Kein Regen Predicted & Label = Kein Regen
    true_pos   += (pred_mask == True)    & (labl_mask == True)
    #              Regen Predicted       & Label = Kein Regen
    false_neg  += (pred_mask == False)   & (labl_mask == True)
    #              Regen Predicted       & Label = Regen
    true_neg   += (pred_mask == False)   & (labl_mask == False)
    #              Kein Regen Predicted  & Label = Regen
    false_pos  += (pred_mask == True)    & (labl_mask == False)
    rain_total += (~labl_mask).sum()
    total += labl_mask.sum() + (~labl_mask).sum()
    
    

    # Simple Baseline
    # aus irgendeinem Grund können wir hier labelToMask nicht nutzen
    pred_simple = simple > 0

    #              Kein Regen Predicted & Label = Kein Regen
    simple_true_pos  += (pred_simple == True)    & (labl_mask == True)
    #              Regen Predicted       & Label = Kein Regen
    simple_false_neg += (pred_simple == False)   & (labl_mask == True)
    #              Regen Predicted       & Label = Regen
    simple_true_neg  += (pred_simple == False)   & (labl_mask == False)
    #              Kein Regen Predicted  & Label = Regen
    simple_false_pos += (pred_simple == True)    & (labl_mask == False)

    return_dict = {
            "TP"        : true_pos.sum(),
            "TN"        : true_neg.sum(),
            "FP"        : false_pos.sum(),
            "FN"        : false_neg.sum(),
            "TP_simple" : simple_true_pos.sum(),
            "TN_simple" : simple_true_neg.sum(),
            "FP_simple" : simple_false_pos.sum(),
            "FN_simple" : simple_false_neg.sum(),
            "total"     : total,
            "rain total":rain_total}


    return return_dict


def sum_up_keys(d):
    for key in d:
        d[key] = np.sum(d[key])
    return d

def add_FPTP_dicts(d1,d2):
    
    for key in d1:
        d1[key] = d1[key] + d2[key]

    return d1

def ZeroInflated_Binomial(t):

    return  tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(tf.stack([1-tf.math.sigmoid(t[...,:1]), tf.math.sigmoid(t[...,:1])],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,:1])),
            tfp.distributions.NegativeBinomial(
            total_count=tf.math.softplus(t[..., 1:2]), 
            logits=tf.math.sigmoid(t[..., 2:]) ),])
        ,name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 )

    


def worker(procNbr, data_path,return_val,dist):
    

    
    processing = 0
    return_dict = {}

    data_x = data_path[0]
    data_y = data_path[1]
    data_p = data_path[2]
    
    threshold_list = np.arange(40+1)

    for i in threshold_list:
        return_dict[i] = None

    while (len(data_x) > 0) :

        x = np.array([data_x.pop()])
        y = np.array([data_y.pop()])
        p = np.array([data_p.pop()])
        
        
        p = dist(p)
        
        for key in threshold_list:
            fptp_dict = get_TP(x[0:,:,:,-1:],y,p,threshold = key/20)
            

            if return_dict[key] is None:
                return_dict[key] = fptp_dict
                continue
            else:
                return_dict[key] = add_FPTP_dicts(return_dict[key],fptp_dict)
        

        processing += 1
        del x
        del y
        del p

    
    del data_x 
    del data_p 
    del data_y
    
    print("Worker {:2d} processed {:6d} images".format(procNbr,processing),end="\r")
    return_val.put(return_dict)
    print("Worker {:2d} finished".format(procNbr))


def save(data,path):
    np.save(path,data,allow_pickle=True)

def load(path):
    return np.load(path,allow_pickle=True)

def multiProc_eval(model,test,getFreshSet,dist=ZeroInflated_Binomial,x_transform=[],y_transform=[]):

    nbrProcesses = cpu_count() * 2
    procCtr = 0
    return_dict = {}
    jobs = []
    finished = False
    shared_dict = {}
    
    train,test = getFreshSet(1)
    d_size = len(test) // nbrProcesses
    batch_size = 200
    data_x = []
    data_y = []
    data_p = []
    train,test = getFreshSet(batch_size)
    j=0
    l = len(test)


    confusionMat = None
    returnQueue = {}
    
    for i,(x,y) in enumerate(test):
        clear_session()
        
        with tf.device("/gpu:0"):
            p = model(x[:,:,:,:],training=False)
        
        
        with tf.device("/cpu:0"):           
            
            for i in range(batch_size):
                if x_transform:
                    new = x[i,:,:,:]
                    for t in x_transform:
                        new_x = t(x[i,:,:,:])
                    data_x.append(new_x)
                else:
                    data_x.append(x[i,:,:,:])
                data_y.append(y[i,:,:,:])
                data_p.append(p[i,:,:,:])
                
            if len(data_x) < d_size:
                continue
        

            
            data = [data_x,data_y,data_p]
            returnQueue[procCtr] = Queue()
        
            job = Process(name = "DeepRain Eval Process "+str(procCtr),
                          target = worker, 
                          args = (procCtr,data,returnQueue[procCtr],dist ))
            procCtr +=1
            
            job.start()
            jobs.append(job)
            data_x = []
            data_y = []
            data_p = []


        while (sleepIfRAMFULL()):
            continue


    def emptyQueue(returnQueue,confusionMat):
        for key in returnQueue:
            threshold_list = returnQueue[key].get()
            
            if confusionMat is None:
                confusionMat = threshold_list
                continue

            for k in threshold_list:
                confusionMat[k] = add_FPTP_dicts(confusionMat[k],threshold_list[k])
        return confusionMat


    confusionMat = emptyQueue(returnQueue,confusionMat)
    for job in jobs[j:]: 
        job.join()
    return confusionMat


def plotAUC(baseline_dict,lw=2):
    figure(num=None, figsize=(20, 30), dpi=100, facecolor='w', edgecolor='k')
    tp = []
    fp = []
    
    for key in baseline_dict:
        base = baseline_dict[key]
        tp.append(base["TP"]/(base["TP"]+base["FN"]) )
        fp.append(base["FP"]/(base["FP"]+base["FN"]))
    sns.set(style="ticks", context="talk")
    sns.set_style("darkgrid")
    plt.style.use("dark_background")
    plt.figure()
    plt.plot(fp, tp, color='darkorange',
         lw=lw, label='ROC curve ')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()