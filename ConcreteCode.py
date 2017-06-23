# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:53:21 2017

@author: Spandan_Mishra
"""

def DynamicTimeWarping(x,y):
    DTW={}
    len1=len(x)
    len2=len(y)
    for i in range(len1):
        DTW[(i,-1)]=float('inf')
    
    for j in range(len2):
        DTW[(-1,j)]=float('inf')
    DTW[(-1,-1)]=0
    for i in range(len1):
        for j in range(len2):
            
            DistTemp= (x[i]-y[j])**2
            DTW[(i,j)]= DistTemp + min(DTW[(i-1,j-1)], DTW[(i-1,j)], DTW[(i,j-1)])
    return(DTW[(len1-1,len2-1)])
    
    
    
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
#%matplotlib notebook

data=[]
TestData=[]
FileNames= glob.glob("*.mat") #
TestFileName=glob.glob("C:/Users/Spandan Mishra/Documents/LambWave/testData/*.mat")

[data.append(loadmat(f,squeeze_me=True, struct_as_record=False)) for f in FileNames]  #loading all the data in the file
[TestData.append(loadmat(f,squeeze_me=True, struct_as_record=False)) for f in TestFileName]
#####################
plt.plot(data[0]['a0'])
plt.plot(data[0]['s0'])
plt.plot(data[1]['s0'])
plt.plot(data[2]['s0'])
plt.plot(data[3]['s0'])
plt.plot(data[4]['s0'])
plt.xlabel('Sampling Number')
plt.ylabel('Volts')

#########################################
frequency=data[0]['setup'].signal_definition.frequency1
# sampling rate of the signal
sampling_rate=data[0]['setup'].sampling_rate
#################################################
########################################### 
 
crosstalk=[]
frequency=[]
sampling_rate=[]
for signals in data:
    frequency.append(signals['setup'].signal_definition.frequency1)
    sampling_rate.append(signals['setup'].sampling_rate)
    crosstalk.append(5/(signals['setup'].signal_definition.frequency1)* (signals['setup'].sampling_rate))
    
##################################################################################
for i in range(5):
    data[i]['s0'][1:int(crosstalk[i])]=0  #training data
    TestData[i]['s0'][1:int(crosstalk[i])]=0  # test data
    
SensorData=list()
ActuatorData=list()

[SensorData.append(signal['s0']) for signal in data ]  # sensor data arranged in list
[ActuatorData.append(signal['a0']) for signal in data]  # Actuator data arrange in list

TestSensorData=list()
TestActuatorData=list()

[TestSensorData.append(signal['s0']) for signal in TestData ]  # Test data  for sensor  arranged in list
[TestActuatorData.append(signal['a0']) for signal in TestData]  # test data for actuator  arrange in list
   
#####################
TotalData=SensorData+TestSensorData
TotalData=np.asarray(TotalData)
  
########################
#Caling Keras
########################
#from tensorflow.examples.tutorials.mnist import input_data
from keras.layers import Input, Dense
from keras.models import Model
#from keras.optimizers import Adam
from keras import regularizers

np.random.seed(7)
SensorData=np.asarray(SensorData) # converting list into array (training data)
TestSensorData=  np.asarray(TestSensorData)
InputSignal=SensorData[0]
actual_signal_len=len(InputSignal)
encoding_dim=100 # This going to be size of our encoded representation
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
autoencoder.fit(TotalData, TotalData, epochs=50, batch_size=1)

#######################
#Encode the images
encoded_signal= encoder.predict(SensorData)
EncodedTestData= encoder.predict(TestSensorData)
"""
In the above code we iterate over the testdata0 to testdata5 to estimate the 
Dynamic time warping value all states with respect to the baseline signal. 
Now a linear relationship between damage size and DTW distance will be used to 
calibrate the damage size.This will enable us to quantify the size of the damage
 as well as its detection. We will set a threshold for the damage detection.

"""
from matplotlib.pyplot import cm
c=cm.rainbow(np.linspace(0,1,5))
labels=['no gap','0.2 mm gap','0.3 mm gap','0.4 mm gap','0.1 mm gap']
#%matplotlib notebook
plt.figure()
for c,signal,l in zip(c,encoded_signal,labels):
    plt.plot(signal,color=c,label=l)
plt.xlabel('Sampling Number')
plt.ylabel('Volts')
plt.legend()

###############################################
c=cm.rainbow(np.linspace(0,1,5))
labels=['no gap','0.2 mm gap','0.3 mm gap','0.4 mm gap','0.1 mm gap']
#%matplotlib notebook
plt.figure()
for c,signal,l in zip(c,EncodedTestData,labels):
    plt.plot(signal,color=c,label=l)

plt.xlabel('Sampling Number')
plt.ylabel('Volts')
plt.legend()
plt.title("Test Data")
################################
# Calling DTW function
DynamicWarp=[]
DynamicWarpTest=[]
[DynamicWarp.append(DynamicTimeWarping(encoded_signal[0],signal)) for signal in encoded_signal]
[DynamicWarpTest.append(DynamicTimeWarping(EncodedTestData[0],signal)) for signal in EncodedTestData]
gaps=[0,0.2,0.3,0.4,0.1]

#linear regression fucntion
#
from sklearn import linear_model
#from sklearn import preprocessing

#DTWScaled=preprocessing.scale(DynamicWarp)

#DTWScaled=np.asarray(DTWScaled)
DynamicWarp=np.asarray(DynamicWarp)
gaps=np.asarray(gaps)
#lets create a regression object
regr= linear_model.LinearRegression(fit_intercept=True,normalize=True)
#fitting the linear regression

regr.fit(DynamicWarp.reshape(5,1),gaps.reshape(5,1))
print("The regression coefficients are as:[%.7f, %.7f]" % (regr.intercept_ ,  regr.coef_))
# The final prediction of the test data
DTWtest=np.asarray(DynamicWarpTest)
###
TrainingGaps=regr.predict(DynamicWarp.reshape(5,1))
PredictionGaps=regr.predict(DTWtest.reshape(5,1))
##########################

plt.figure()
plt.scatter(gaps,PredictionGaps, c='b', marker="s")
plt.xlabel('Actual Gap')
plt.ylabel('predicted Gaps')
plt.show()




