# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:43:56 2017

@author: Spandan_Mishra
"""


from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

class ReturnValues:
    # this Object will be used to return the output
    def __init__(self, data0, data1,data2,data3):
        self.data0 =  data0
        self.data1 =  data1
        self.data2 =  data2
        self.data3 =   data3

def loadData(FilePath):    
    NewTrainOne=[]
    ####################################
    NewTrainingData1=[]
    
    def key_func(x):
            return os.path.split(x)[-1]
    for data_file1 in sorted(os.listdir(FilePath),key=key_func):
        NewTrainingData1.append(data_file1)
    #########################################
    #folderPath="C:/Users/Spandan Mishra/Documents/GitHub/LambWave/Metal1"
    [NewTrainOne.append(loadmat(os.path.join(FilePath,f),squeeze_me=True, struct_as_record=False)) for f in NewTrainingData1]
    ###############################################
    frequency=NewTrainOne[4]['setup'].signal_definition.frequency1
    # sampling rate of the signal
    sampling_rate=NewTrainOne[0]['setup'].sampling_rate
    ##################################################
    crosstalk=[]
    frequency=[]
    sampling_rate=[]
    for signals in NewTrainOne:
        frequency.append(signals['setup'].signal_definition.frequency1)
        sampling_rate.append(signals['setup'].sampling_rate)
        crosstalk.append(5/(signals['setup'].signal_definition.frequency1)* (signals['setup'].sampling_rate))
    ######################################
    for i in range(25):
        NewTrainOne[i]['s0'][1:int(crosstalk[i])]=0  #training data
        
    SensorData=list()
    ActuatorData=list()
    
    [SensorData.append(signal['s0']) for signal in NewTrainOne ]  # sensor data arranged in list
    [ActuatorData.append(signal['a0']) for signal in NewTrainOne]  # Actuator data arrange in list
    
    return ReturnValues(SensorData,ActuatorData,sampling_rate,frequency)
###########################################################
def SparseEncoder(folder):
    Workpath="C:/Users/Spandan Mishra/Documents/GitHub/LambWave/relambwaveresultonconcrete/"+folder
    LambData= loadData(Workpath)
    TotalData=LambData.data0
    SensorData=LambData.data0
    TotalData=np.asarray(TotalData)
    #####################################
    #####################################
    from keras.layers import Input, Dense
    from keras.models import Model
    from keras import regularizers
    
    np.random.seed(7)
    SensorData=np.asarray(SensorData) # converting list into array (training data)
    InputSignal=SensorData[0]
    actual_signal_len=len(InputSignal)
    encoding_dim=500 # This going to be size of our encoded representation
    #this returns a tensor
    inputs = Input(shape=(actual_signal_len,))
    encoded=Dense(encoding_dim,activation='relu',activity_regularizer=regularizers.l1(10e-5))(inputs)
    decoded=Dense(actual_signal_len)(encoded)
    
    # this model maps an input to its encoded representation
    encoder = Model(input=inputs, output=encoded)
    
    ## this model maps an input to its reconstruction
    autoencoder= Model(input = inputs, output= decoded)
    #we'll configure our model to use a mean squarred error loss, and the Adam optimizer
    # we also train autoencoder for 50 epochs
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(TotalData, TotalData, epochs=100, batch_size=1)
    #######################
    #Encode the images
    encoded_signal= encoder.predict(SensorData)
    ################################
    return encoded_signal
    
    ################################
    
class ReturnFinalValues:
# this Object will be used to return the output
    def __init__(self, data0, data1):
        self.data0 =  data0
        self.data1 =  data1
        
def CalibrationModel(input_file=None):  
    from sklearn import linear_model
    from math import log
    from numpy import exp
    import numpy as np
    
    if input_file is None:
        inputFilePath="C:/Users/Spandan Mishra/Documents/GitHub/LambWave/"
        input_file="ConcreteFileName.txt"
    input_file=inputFilePath+input_file						# This is to check if no input was passed as input python function

    with open(input_file,"r") as fileReader:
    	folder=fileReader.read()

    encoded_signal=SparseEncoder(folder)
    dist1=[]
    
    def mean(a):
        return sum(a)/len(a)
    MeanBaseline=[]
    
    [MeanBaseline.append(i) for i in    map(mean,zip(*encoded_signal[0:4]))]
    
    
    for x in encoded_signal:
        dist1.append(np.linalg.norm(MeanBaseline-x))
        
    ############################################
    SelectedDist=[]
    
    index=[0,5,10,15,20]
    [SelectedDist.append(dist1[i]) for i in index]
       
    ###############################################
   
    gaps=[0.00001,0.1,0.2,0.3,0.4]
    logGap=[]
    [logGap.append(log(i))  for i in gaps]
    SelectedDist=np.asarray(SelectedDist)
    logGap=np.asarray(logGap)
    
    expRegr=linear_model.LinearRegression(fit_intercept=True,normalize=True)
    expRegr.fit(SelectedDist.reshape(5,1),logGap.reshape(5,1))
    print("The regression coefficients are as:[%.7f, %.7f]" % (expRegr.intercept_ ,  expRegr.coef_))
    #TrainingGapsExp=expRegr.predict(np.asarray(SelectedDist).reshape(5,1))
    #PredictedGap=exp(TrainingGapsExp)
    testData=np.asarray(dist1)
    Pred=expRegr.predict(testData.reshape(25,1)) # predeiction of the Metal 1
    PredictedGap=exp(Pred)
    #x=[0.00001,0.1,0.2,0.3]
    GapVec=[]
    [GapVec.append(np.tile(i,(1,5))) for i in gaps]
    GapVecNew = np.concatenate([np.array(i[0]) for i in GapVec])  # convert gapvec in numpy
    error=np.mean(np.square(np.subtract(GapVecNew,PredictedGap)))
    outputFileName="ConcreteOutput"+folder+".txt"

    np.savetxt(outputFileName, GapVecNew)
        

    return ReturnFinalValues(PredictedGap,error)

