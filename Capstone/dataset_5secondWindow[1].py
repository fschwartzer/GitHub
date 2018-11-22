# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:14:55 2018

@author: fernando.schwartzer
"""

import numpy as np 
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("dataset_5secondWindow[1].csv")

print(data.target.unique())
print("----------------------------------------")
print(data.target.value_counts())

sns.set(rc={'figure.figsize':(13,6)})
fig = sns.countplot(x = "target" , data = data)
plt.xlabel("Mode")
plt.ylabel("Count")
plt.title("Mode Count")
plt.grid(True)
plt.show(fig)

import seaborn as sns
sns.pairplot(data, hue='target', size=3)

# Dividindo os dados entre features e coluna alvo
target_raw = data['target']
features_raw = data.drop('target', axis=1)


# Importando sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Inicializando um aplicador de escala e aplicando em seguida aos atributos
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['time', 'android.sensor.accelerometer#mean', 'android.sensor.accelerometer#min', 'android.sensor.accelerometer#max', 'android.sensor.accelerometer#std', 'android.sensor.gyroscope#mean', 'android.sensor.gyroscope#min', 'android.sensor.gyroscope#max', 'android.sensor.gyroscope#std', 'sound#mean', 'sound#min', 'sound#max', 'sound#std']

features_minmax_transform = pd.DataFrame(data = features_raw)
features_minmax_transform[numerical] = scaler.fit_transform(features_raw[numerical])

# Exibindo um exemplo de registro com a escala aplicada
display(features_minmax_transform.head(n=5))


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(target_raw)
target_le = le.transform(target_raw)

# Importar train_test_split
from sklearn.cross_validation import train_test_split

# Dividir os 'atributos' e 'income' entre conjuntos de treinamento e de testes.
X_train, X_test, y_train, y_test = train_test_split(features_minmax_transform, 
                                                    target_le, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] =  accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta =0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta =0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

