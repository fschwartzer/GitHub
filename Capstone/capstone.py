# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:38:28 2018

@author: fernando.schwartzer
"""


import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset_5secondWindow.csv', sep=',')
dataset = dataset.drop(['Unnamed: 0', 'id', 'user'], axis=1)
dataset.fillna(0, inplace=True)
dataset.T

print(dataset.target.unique())
print("----------------------------------------")
print(dataset.target.value_counts())

sns.set(rc={'figure.figsize':(13,6)})
fig = sns.countplot(x = "target" , data = dataset)
plt.xlabel("Mode")
plt.ylabel("Count")
plt.title("Mode Count")
plt.grid(True)
plt.show(fig)


sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='target', y= 'android.sensor.accelerometer#mean', data= dataset, jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)

sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='target', y= 'android.sensor.gyroscope#mean', data= dataset, jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)

sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='target', y= 'android.sensor.linear_acceleration#mean', data= dataset, jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)

target = dataset['target']
features = dataset.drop('target', axis = 1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(features)
features_scaled = scaler.transform(features)


x_train ,x_test = train_test_split(features_scaled,test_size=0.2) 

