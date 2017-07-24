# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 22:21:23 2017

@author: Spandan_Mishra
"""

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

NewTrainOne=[]
NewTrainTwo=[]
####################################
NewTrainingData1=[]
NewTrainingData2=[]
def key_func(x):
        return os.path.split(x)[-1]
for data_file1,data_file2 in zip(sorted(os.listdir("Metal1/"),key=key_func),sorted(os.listdir("Metal2/"),key=key_func)):
    NewTrainingData1.append(data_file1)
    NewTrainingData2.append(data_file2)

#########################################
folderPath="C:/Users/Spandan Mishra/Documents/GitHub/LambWave/Metal1"
[NewTrainOne.append(loadmat(os.path.join(folderPath,f),squeeze_me=True, struct_as_record=False)) for f in NewTrainingData1]
[NewTrainTwo.append(loadmat(os.path.join(folderPath,f),squeeze_me=True, struct_as_record=False)) for f in NewTrainingData2]
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
    NewTrainTwo[i]['s0'][1:int(crosstalk[i])]=0  # test data
    
SensorData=list()
ActuatorData=list()

[SensorData.append(signal['s0']) for signal in NewTrainOne ]  # sensor data arranged in list
[ActuatorData.append(signal['a0']) for signal in NewTrainOne]  # Actuator data arrange in list
##############################################################################################################
TestSensorData=list()
TestActuatorData=list()

[TestSensorData.append(signal['s0']) for signal in NewTrainTwo ]  # Test data  for sensor  arranged in list
[TestActuatorData.append(signal['a0']) for signal in NewTrainTwo]  # test data for actuator  arrange in list
###########################################################
TotalData=SensorData+TestSensorData
TotalData=np.asarray(TotalData)
#####################################
#####################################
from keras.layers import Input, Dense
from keras.models import Model
#from keras.optimizers import Adam
from keras import regularizers

np.random.seed(7)
SensorData=np.asarray(SensorData) # converting list into array (training data)
TestSensorData=  np.asarray(TestSensorData)
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
EncodedTestData= encoder.predict(TestSensorData)
################################
dist1=[]
dist2=[]
def mean(a):
    return sum(a)/len(a)
MeanBaseline=[]
MeanBaselineTest=[]
[MeanBaseline.append(i) for i in    map(mean,zip(*encoded_signal[0:4]))]
[MeanBaselineTest.append(i) for i in    map(mean,zip(*EncodedTestData[0:4]))]

for x,y in zip(encoded_signal,EncodedTestData):
    dist1.append(np.linalg.norm(MeanBaseline-x))
    dist2.append(np.linalg.norm(EncodedTestData[0]-y))
############################################
SelectedDist=[]
SelectedDistTest=[]  # test data
index=[0,5,10,15]
[SelectedDist.append(dist1[i]) for i in index]
[SelectedDistTest.append(dist2[i]) for i in index]
###############################################
from sklearn import linear_model
from math import log
from numpy import exp
gaps=[0.00001,0.1,0.2,0.3]
logGap=[]
[logGap.append(log(i))  for i in gaps]
SelectedDist=np.asarray(SelectedDist)
logGap=np.asarray(logGap)

expRegr=linear_model.LinearRegression(fit_intercept=True,normalize=True)
expRegr.fit(SelectedDist.reshape(4,1),logGap.reshape(4,1))
print("The regression coefficients are as:[%.7f, %.7f]" % (expRegr.intercept_ ,  expRegr.coef_))
TrainingGapsExp=expRegr.predict(np.asarray(SelectedDist).reshape(4,1))
#######################################################
testData=np.asarray(dist1)
Pred=expRegr.predict(testData[:-5].reshape(20,1)) # predeiction of the Metal 1
PredictedGap=exp(Pred)
print(PredictedGap)
#######################################################
x=[0.00001,0.1,0.2,0.3]
GapVec=[]
[GapVec.append(np.tile(i,(1,5))) for i in x]
plt.figure()
plt.scatter(SelectedDist,gaps,color='blue',label='Data used to Train')
plt.plot(SelectedDist,exp(TrainingGapsExp),color='red',label='Prediction on Training data')
plt.scatter(testData[:-5],PredictedGap,color='green',label='Prediction on Metal 1 Data')
plt.ylabel('Gaps (mm)')
plt.xlabel('Damage Index')
plt.legend()
plt.show()
#####################################
d=[0,0.1,0.2,0.3,0.4]
plt.figure()
plt.scatter(GapVec,PredictedGap)
plt.plot(d,d,color='red')
plt.xlabel('Actual Damage Size')
plt.ylabel('Predicted Damage Size')
plt.show()

###################
##prediction error
GapVecNew = np.concatenate([np.array(i[0]) for i in GapVec])  # convert gapvec in numpy

error=np.sum(np.square(np.subtract(GapVecNew,PredictedGap)))

### Data from Metal 2##########
xMetal2=[0.00001,0.1,0.2,0.3,0.4]
GapVecMetal2=[]
[GapVecMetal2.append(np.tile(i,(1,5))) for i in xMetal2]
testDataMetal2=np.asarray(dist2)
PredictedValueMetal2=expRegr.predict(testDataMetal2.reshape(25,1))
print(exp(PredictedValueMetal2))

plt.figure()
plt.scatter(GapVecMetal2,exp(PredictedValueMetal2),color='red',label='Prediction on Metal 2 data')
plt.ylabel('Gaps (mm)')
plt.xlabel('Damage Index')
plt.legend()
plt.show()