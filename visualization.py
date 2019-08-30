#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:25:52 2019

@author: admin
"""
from datetime import datetime
from pandas import read_csv
from matplotlib import pyplot
import seaborn as sns
sns.set_style()

# load dataset
dataset = read_csv('/Users/admin/Desktop/Project/2019 wether/Beijing/code/pollution.csv', header=0, index_col=0)
values = dataset.values
print(dataset.corr())

# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()


dataset.columns = ['PM25', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snowfall', 'rainfall']
dataset.index.name = 'date'
co_data=dataset[['PM25','dew','temp','press','wnd_spd','snowfall','rainfall']]
g=sns.PairGrid(co_data)
g.map(pyplot.scatter)
def plot_corr_map(df):
    corr=df.corr()
    _,ax=pyplot.subplots(figsize=(12,10))
    cmap=sns.diverging_palette(220,10,as_cmap=True)
    _=sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink':0.9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize':12})
plot_corr_map(co_data)

