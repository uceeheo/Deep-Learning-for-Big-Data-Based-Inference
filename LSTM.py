##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Wed Jun 19 16:55:14 2019
#
#@author: admin
#"""
from keras.models import Model
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from keras.layers.recurrent import SimpleRNN
import numpy as np
from keras import layers
from math import sqrt
from numpy import concatenate    
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,GRU, RNN,Dropout
from keras.layers import LSTM
from keras.regularizers import l2
from keras import regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import seaborn as sns
sns.set_style()
import pandas as pd
import time
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)# 只考虑当前时刻(t)的前一时刻（t-1）的PM2.5值
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 96
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#
#model = Sequential()
#model.add(SimpleRNN(32, input_shape=(train_X.shape[1], train_X.shape[2]),kernel_regularizer=l2(0.005),
#               recurrent_regularizer=l2(0.005)))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
#print (model.summary())
#
model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]),kernel_regularizer=l2(0.005),
               recurrent_regularizer=l2(0.005)))
model.add(Dense(1)) # Dense just means a fully connected layer, the parameter is the number of neurons in that layer.
model.compile(loss='mae', optimizer='adam')
print (model.summary())

#
# fit network
#from keras.callbacks import EarlyStopping
#earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)
#history=model.fit(train_X,train_y,epochs=100,batch_size=48,validation_data=(test_X,test_y) ,verbose=2,callbacks=[earlyStop])

history = model.fit(train_X, train_y, epochs=50,batch_size=48, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
##
# make a prediction 为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
start=time.time()
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast 
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)# 数组拼接 这里注意的是保持拼接后的数组  列数  需要与之前的保持一致
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual         
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)  # 将标准化的数据转化为原来的范围
inv_y = inv_y[:,0]
end = time.time()
#print('This took {} seconds.'.format(end - start))
#
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
r2 = metrics.r2_score(inv_y, inv_yhat)
print('R2:%f' % r2)
print('Test RMSE: %.3f' % rmse)

pyplot.plot(np.array(inv_y)[-48:], label='True Data')
pyplot.plot(np.array(inv_yhat)[-48:], label='Prediction')
pyplot.legend()
pyplot.show()
rmse = sqrt(mean_squared_error(inv_y[-48:] ,(inv_yhat)[-48:]))
print('Prediction RMSE: %.3f' % rmse)
# 
#
#
#
#
#
# multi lag 
# specify the number of lag hours 
#n_hours = 1
#n_features = 8
## frame as supervised learning
#reframed = series_to_supervised(scaled, n_hours, 1)
#print(reframed.shape)
# 
## split into train and test sets
#values = reframed.values
#n_train_hours = 365 * 96
#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]
## split into input and outputs
#n_obs = n_hours * n_features
#train_X, train_y = train[:, :n_obs], train[:, -n_features]
#test_X, test_y = test[:, :n_obs], test[:, -n_features]
#print(train_X.shape, len(train_X), train_y.shape)
#
## reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
#test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# 
# design network
#model = Sequential()
#model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]),kernel_regularizer=l2(0.005),
#               recurrent_regularizer=l2(0.005)))
#model.add(Dropout(0.1))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
# fit network
#history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2)
## plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
#
## make a prediction
#yhat = model.predict(test_X)
#test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
## invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:,0]
## invert scaling for actual
#test_y = test_y.reshape((len(test_y), 1))
#inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,0]
## calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#r2 = metrics.r2_score(inv_y, inv_yhat)
#print('R2:%f' % r2)
#print('Test RMSE: %.3f' % rmse)
##
#pyplot.plot(np.array(inv_y)[-24:], label='True Data')
#pyplot.plot(np.array(inv_yhat)[-24:], label='Prediction')
#pyplot.legend()
#pyplot.show()
#rmse = sqrt(mean_squared_error(inv_y[-24:] ,(inv_yhat)[-24:]))
#print('Prediction RMSE: %.3f' % rmse)



# failure trying
# rnn

# design network 
#model = Sequential()
#model.add(layers.Flatten(input_shape=(train_X.shape[1:])))
#model.add(layers.Dense(32,activation='relu'))
#model.add(layers.Dense(1))
#model.compile(loss='mae', optimizer='adam')
#print (model.summary())
# stacked lstm
#model = Sequential()
#model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.005),
#               recurrent_regularizer=l2(0.005)))
#model.add(Dropout(0.2))
#model.add(LSTM(32, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005)))
#model.add(Dropout(0.2))
#model.add(Dense(1)) #Dense just means a fully connected layer, the parameter is the number of neurons in that layer.
#model.compile(loss='mae', optimizer='adam')
#print (model.summary())
#model = Sequential()
#model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(32))
#model.add(Dropout(0.2))
#model.add(Dense(1))
#model.compile(loss='mae', optimizer='adam')
#print (model.summary())
## GRU
##model.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2])))

## stacked lstm
##model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
##model.add(Dropout(0.2))
##model.add(LSTM(20, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005)))
##model.add(Dropout(0.2))

#model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, kernel_regularizer=l2(0.005),
#               recurrent_regularizer=l2(0.005)))
#model.add(LSTM(32, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005)))
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='linear'))
#model.compile(loss='mae', optimizer='adam')
#print (model.summary())


## bidirectional model
##from keras.layers import Bidirectional
##model = Sequential()
##model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(train_X.shape[1], train_X.shape[2])))

##model = Sequential()
##model.add(LSTM(27, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
##model.add(Dropout(0.2))
##model.add(LSTM(90, return_sequences=True))
##model.add(Dropout(0.2))
##model.add(LSTM(47))
##model.add(Dropout(0.2))
##model.add(Dense(1))
##model.compile(loss='mae', optimizer='adam')

## invert scaling for actual #wrong
##inv_y = scaler.inverse_transform(test_X)
##inv_y = inv_y[:,0]
