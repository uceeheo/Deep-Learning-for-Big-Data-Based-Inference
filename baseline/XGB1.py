#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:02:04 2019

@author: admin
"""
from math import sqrt
import pandas as pd
from xgboost.sklearn import XGBRegressor as xgbr
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt  
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_absolute_error as MSE

file =pd.read_csv("/Users/admin/Desktop/Project/2019 wether/Beijing/code/PRSA_data_2010.1.1-2014.12.31.csv")

test=file[35064:] 
train=file[:35064] 
train=train.dropna()
test=test.dropna()
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#ytrain=train["pm2.5"]
#xtrain=train[['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES',
#       'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]
#ytest=test["pm2.5"]
#xtest=test[['year', 'month','day', 'hour', 'DEWP', 'TEMP', 'PRES',
#       'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]

ytrain=train["pm2.5"]
xtrain=train[['year', 'month', 'day', 'hour', 'DEWP', 'TEMP',
              'Iws', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]
ytest=test["pm2.5"]
xtest=test[['year', 'month', 'day', 'hour', 'DEWP', 'TEMP',
            'Iws', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']]

model_xgb = xgbr(max_depth=8, learning_rate=0.1, n_estimators=200, silent=True, objective='reg:gamma')
model_xgb.fit(xtrain,ytrain)

print("score:",model_xgb.score(xtest,ytest))
print("mae:",mean_absolute_error(ytest ,model_xgb.predict(xtest)))
print("mse:",mean_squared_error(ytest ,model_xgb.predict(xtest)))
rmse = sqrt(mean_squared_error(ytest ,model_xgb.predict(xtest)))
print('Test RMSE: %.3f' % rmse)
print("r2:",r2_score(ytest ,model_xgb.predict(xtest)))
print("ymean:",ytest.mean())

pyplot.plot(np.array(ytest)[-24:], label='True Data')
pyplot.plot(model_xgb.predict(xtest)[-24:], label='Prediction')
pyplot.legend()
pyplot.show()
rmse = sqrt(mean_squared_error(ytest[-24:] ,model_xgb.predict(xtest)[-24:]))
print('Prediction RMSE: %.3f' % rmse)

x=xtrain.columns
y=model_xgb.feature_importances_
fig = plt.figure(figsize=(13,5))
plt.bar(x,y,0.4)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature_importances")
plt.show()

