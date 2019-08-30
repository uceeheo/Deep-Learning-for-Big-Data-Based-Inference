#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:49:14 2019

@author: admin
"""
from math import sqrt
import pandas as pd
from xgboost.sklearn import XGBRegressor as xgbr
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt  # 导入图形展示库
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_absolute_error as MSE

file =pd.read_csv("/Users/admin/Desktop/Project/2019 wether/Beijing/code/PRSA_data_2010.1.1-2014.12.31.csv")

test=file[35064:] #2014年的数据
train=file[:35064] #2014年以前的数据
train=train.dropna()
test=test.dropna()
train=pd.get_dummies(train)
test=pd.get_dummies(test)

ytrain=train["pm2.5"]
xtrain=train[['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES',
       'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv',"pm2.5_1ago","pm2.5_2ago","pm2.5_3ago"]]
ytest=test["pm2.5"]
xtest=test[['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES',
       'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv',"pm2.5_1ago","pm2.5_2ago","pm2.5_3ago"]]

model_xgb2 = xgbr(max_depth=8, learning_rate=0.1, n_estimators=200, silent=True, objective='reg:gamma')
model_xgb2.fit(xtrain,ytrain)
print("score:",model_xgb2.score(xtest,ytest))
print("MSE:",MSE(ytest,model_xgb2.predict(xtest)))
rmse = sqrt(mean_squared_error(ytest ,model_xgb2.predict(xtest)))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(np.array(ytest)[-24:], label='True Data')
pyplot.plot(model_xgb2.predict(xtest)[-24:], label='Prediction')
pyplot.legend()
pyplot.show()
rmse = sqrt(mean_squared_error(ytest[-24:] ,model_xgb2.predict(xtest)[-24:]))
print('Prediction RMSE: %.3f' % rmse)

x=xtrain.columns
y=model_xgb2.feature_importances_
fig = plt.figure(figsize=(20,5))
plt.bar(x,y,0.4)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature_importances")
plt.show()


