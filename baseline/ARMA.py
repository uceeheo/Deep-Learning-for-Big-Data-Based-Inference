import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from statsmodels.tsa import stattools
from pandas import concat
from pandas import DataFrame
# Configure visualisations
sns.set_style()
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
from statsmodels.graphics.api import qqplot


data=pd.read_csv('/Users/admin/Desktop/Project/2019 wether/Beijing/code/pollution.csv')
data.columns
data.head()
data.tail()
data.info()


plt.figure(figsize=(16,8))
plt.plot(data['PM25'])
plt.title('Trend of PM2.5')
plt.ylabel('PM2.5')
plt.legend(loc='best')
plt.show()

# Stationarity test
from statsmodels.tsa.stattools import adfuller as ADF
print(ADF(data.PM25))
diff=0
adf=ADF(data['PM25'])
while adf[1]>0.05:
    diff=diff+1
    adf=ADF(data['PM2.5'].diff(diff).dropna())
print('The original sequence tends to be smoothed after {}th order difference，p={}'.format(diff, adf[1]))

# White noise test
from statsmodels.stats.diagnostic import acorr_ljungbox
[[lb], [p]] = acorr_ljungbox(data['PM25'], lags = 1)
if p < 0.05:
    print(u'The original sequence is a non-white noise sequence，p=%s' %p)
else:
    print(u'The original sequence is a white noise sequence，p=%s' %p)
[[lb], [p]] = acorr_ljungbox(data['PM25'].diff().dropna(), lags = 1)
if p < 0.05:
    print(u'First order difference sequence is a non-white noise sequence，p=%s' %p)
else:
    print(u'First order difference sequence is a white noise sequence，p=%s' %p)
    
# ACF,PACF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
def plot_acfandpacf(dataset):
    data = dataset.PM25
    fig = plt.figure(figsize=(40,10))
    ax1=fig.add_subplot(211)
    plot_acf(data,lags=40,ax=ax1).show()
    ax2=fig.add_subplot(212)
    plot_pacf(data,lags=40,ax=ax2).show()
    plt.show()  
#    
plot_acfandpacf(data)
    

import statsmodels.tsa.stattools as st
order= st.s(data.PM25,max_ar=3,max_ma=3,ic=['aic','bic','hqic'])
print(order.bic_min_order)

def plot_results(predicted_data, true_data):
  fig = plt.figure(facecolor='white',figsize=(10,5))
  ax = fig.add_subplot(111)
  ax.plot(true_data, label='True Data')
  plt.plot(predicted_data, label='Prediction')
  plt.legend()
  plt.show()
def arma_predict(dataset,number):
  data = list(dataset.PM25)
  from statsmodels.tsa.arima_model import ARMA

  model = ARMA(data, order=(3,2))
  result_arma = model.fit(disp=-1, method='css')
  predict = result_arma.predict(len(data)-number,len(data))
  RMSE = np.sqrt(((predict-data[len(data)-number-1:])**2).sum()/(number+1))
  plot_results(predict,data[len(data)-number-1:])
  return predict,RMSE

predict,RMSE=arma_predict(data,23)
print(RMSE)



# qq plot
#from scipy import stats
#from statsmodels.tsa.arima_model import ARMA
#data = list(data.PM25)
#model = ARMA(data, order=(3,2))
#result = model.fit(disp=-1, method='css')
#stats.normaltest(result.resid)
#fig=plt.figure(figsize=(6,6))
#print(result.summary())
#ax=fig.add_subplot(111)
#fig=qqplot(result.resid,line='q',ax=ax,fit=True)
#plt.title("Normal Q-Q plot")
#plt.show()
#stats.normaltest(result.resid)
#resid = result.resid
#plt.hist(resid, # 绘图数据
#        bins = 100, # 指定直方图条的个数
#        normed = True, # 设置为频率直方图
#        color = 'steelblue', # 指定填充色
#        edgecolor = 'k') # 指定直方图的边界色
#plt.title('Histogram of residuals')
#from statsmodels.stats.stattools import durbin_watson
#print(durbin_watson(result.resid))


#plot_results(predict,data['PM25'])


