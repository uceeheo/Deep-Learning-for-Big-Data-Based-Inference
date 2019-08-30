#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 12:42:51 2019

@author: admin
"""

import pandas as pd
import numpy as np

#清洗数据：读取并删除空值
pm = pd.read_csv('/Users/admin/Desktop/Project/2019 wether/Beijing/dataframe/raw.csv')
pm.dropna(inplace=True)

#定义空气质量等级
def get_grade(value):
   if value <= 50 and value>=0:
      return 'Good'
   elif value <= 100:
      return 'Moderate' 
   elif value <= 150:
      return 'Unhealthy for Sensi'
   elif value <= 200:
      return 'Unhealthy'
   elif value <= 300:
      return 'Very Unhealthy'
   elif value <= 500:
      return 'Hazardous' 
   elif value > 500:
      return 'Beyond Index' 
   else:
      return None 

#增加空气质量’Grade’字段
pm.loc[:, 'Grade'] = pm['pm2.5'].apply(get_grade)

#定义各年份空气质量
pm2010 = pm[pm['year'] == 2010]
pm2011 = pm[pm['year'] == 2011]
pm2012 = pm[pm['year'] == 2012]
pm2013 = pm[pm['year'] == 2013]
pm2014 = pm[pm['year'] == 2014]

#分别统计各年份各个空气质量等级占比天数
grade2010 = pm2010.groupby(['Grade']).size()/len(pm2010)
grade2011 = pm2011.groupby(['Grade']).size()/len(pm2011)
grade2012 = pm2012.groupby(['Grade']).size()/len(pm2012)
grade2013 = pm2013.groupby(['Grade']).size()/len(pm2013)
grade2014 = pm2014.groupby(['Grade']).size()/len(pm2014)

#定义一个空气质量等级索引
ix_grade = ['Good', 'Moderate', 'Unhealthy for Sensi', 'Unhealthy', 'Very Unhealthy', 'Hazardous','Beyond Index']

#创建一个DataFrame，把2010年的空气质量等级占比数据加进去，把其他三个等级数据也加进去
df_year = pd.DataFrame(grade2010, index = ix_grade, columns=['2010'])
df_year['2011'] = grade2011
df_year['2012'] = grade2012
df_year['2013'] = grade2013
df_year['2014'] = grade2014

#输出年级别空气质量，并作图
print(df_year)
#df_year.ix[:,['2010','2011',‘'2012','2013','2014']].plot.bar(title='AQI 2010 - 2014', figsize=(8,6), fontsize=12)

#计算当地五年的pm2.5测量值月度平均值
month2010 = pm2010.groupby(['month'])['pm2.5'].mean()
month2011 = pm2011.groupby(['month'])['pm2.5'].mean()
month2012 = pm2012.groupby(['month'])['pm2.5'].mean()
month2013 = pm2013.groupby(['month'])['pm2.5'].mean()
month2014 = pm2014.groupby(['month'])['pm2.5'].mean()

#合并数据
df_month = pd.DataFrame({'2010':month2010}, index = np.arange(1,13))
df_month['2011'] = month2011
df_month['2012'] = month2012
df_month['2013'] = month2013
df_month['2014'] = month2014

#输出数据并作图
print(df_month)
#df_month.ix[:, ['2010','2011',‘'2012','2013','2014']].plot(title='PM2.5 Monthly Avg. 2010 - 2014', figsize=(8,4)

#定义每小时的pm2.5，并输出各年小时级别空气质量对比数据，并画出对比图
df_hour = pd.DataFrame({'month': pm2010.loc[:,'month'],
'day' : pm2010.loc[:,'day'],
'hour' : pm2010.loc[:,'hour'],
'2010':pm2010.loc[:,'pm2.5']})

#拼接数据
df_hour = df_hour.merge(pm2011.loc[:,['month','day','hour','pm2.5']], on=('month','day','hour'))
df_hour.rename({'pm2.5':'2011'}, axis="columns", inplace=True)

df_hour = df_hour.merge(pm2012.loc[:,['month','day','hour','pm2.5']], on=('month','day','hour'))
df_hour.rename({'pm2.5':'2012'}, axis="columns", inplace=True)

df_hour = df_hour.merge(pm2013.loc[:,['month','day','hour','pm2.5']], on=('month','day','hour'))
df_hour.rename({'pm2.5':'2012'}, axis="columns", inplace=True)

df_hour = df_hour.merge(pm2014.loc[:,['month','day','hour','pm2.5']], on=('month','day','hour'))
df_hour.rename({'pm2.5':'2012'}, axis="columns", inplace=True)

#各年与前一年小时级别空气质量对比
#greater_2011and2010 = len(df_hour[df_hour['2011']>df_hour['2010']]), 
#1.0/len(df_hour[df_hour['2011']>df_hour['2010']])/len(df_hour)
#less_2011and2010 = len(df_hour[df_hour['2011']<df_hour['2010']]),
#1.0/len(df_hour[df_hour['2011']<df_hour['2010']])/len(df_hour)
#
#greater_2012and2011 = len(df_hour[df_hour['2012']>df_hour['2011']]), 
#1.0/len(df_hour[df_hour['2012']>df_hour['2011']])/len(df_hour)
#less_2012and2011 = len(df_hour[df_hour['2012']<df_hour['2011']]),
#1.0/len(df_hour[df_hour['2012']<df_hour['2011']])/len(df_hour)
#
#greater_2013and2012 = len(df_hour[df_hour['2013']>df_hour['2012']]), 
#1.0/len(df_hour[df_hour['2013']>df_hour['2012']])/len(df_hour)
#less_2013and2012 = len(df_hour[df_hour['2013']<df_hour['2012']]),
#1.0/len(df_hour[df_hour['2013']<df_hour['2012']])/len(df_hour)
#
#greater_2014and2013 = len(df_hour[df_hour['2014']>df_hour['2013']]), 
#1.0/len(df_hour[df_hour['2014']>df_hour['2013']])/len(df_hour)
#less_2014and2013 = len(df_hour[df_hour['2014']<df_hour['2013']]),
#1.0/len(df_hour[df_hour['2014']<df_hour['2013']])/len(df_hour)

#输出结果
#print(greater_2011and2010,less_2011and2010)
#print(greater_2012and2011,less_2012and2011)
#print(greater_2013and2012,less_2013and2012)
#print(greater_2014and2013,less_2014and2013)
