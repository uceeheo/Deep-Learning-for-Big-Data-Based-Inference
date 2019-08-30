#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:13:00 2019

@author: admin
"""
import pandas as pd
from pandas import read_csv
from datetime import datetime
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

dataset = read_csv('/Users/admin/Desktop/Project/2019 wether/Beijing/dataframe/raw.csv')
print(dataset.describe())
#count   #数量
#mean    #均值
#std     #标准差
#min     #最小值
#25%     #下四分位
#50%     #中位数
#75%     #上四分位
#max     #最大值
print(len(dataset["pm2.5"][pd.isnull(dataset["pm2.5"])])/len(dataset))
print(dataset["pm2.5"].isnull().sum())
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('/Users/admin/Desktop/Project/2019 wether/Beijing/dataframe/raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)


dataset.columns = ['PM25', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snowfall', 'rainfall']
dataset.index.name = 'date'

dataset['PM25'].fillna(0, inplace=True)

dataset = dataset[24:]

print(dataset.head(5))

dataset.to_csv('pollution.csv')