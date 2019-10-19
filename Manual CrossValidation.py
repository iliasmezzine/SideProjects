#!/usr/bin/env python
# coding: utf-8

# In[207]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors, metrics
import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('winequality-white.csv', sep=";")
data["quality"] = np.where(data.quality<6, 0, 1)

def concat(lst):
    return pd.concat(lst,axis=0)


def chunks(data,n_folds):
    spl = np.array_split(data,n_folds)
    return [[spl[i],concat([spl[j] for j in range(len(spl)) if j != i])] for i in range(len(spl))]
             

def CV_score(p,chunk_list):
    
    Classifier = neighbors.KNeighborsClassifier(n_neighbors=p)
    Scaler = preprocessing.StandardScaler()
    
    scores = []
    
    for test,train in chunk_list:
        
        X_train = Scaler.fit_transform(train.as_matrix(train.columns[:-1]))
        X_test = Scaler.fit_transform(test.as_matrix(test.columns[:-1]))
        y_train = train.as_matrix([train.columns[-1]]).flatten()
        y_test = test.as_matrix([test.columns[-1]]).flatten()
        
        Classifier.fit(X_train,y_train)
        y_pred = Classifier.predict(X_test)
        scores.append(metrics.accuracy_score(y_pred,y_test))
        
    return np.mean(scores)

def kNNGridSearch(dataset,n_folds,params_grid):
    chunk_list = chunks(dataset,n_folds)
    mean_accuracy_scores = [(p,CV_score(p,chunk_list)) for p in params_grid]
    return mean_accuracy_scores


# In[211]:


params_grid = [i+1 for i in range(50)]
dataset = data
n_folds = 5

kNNGridSearch(dataset,n_folds,params_grid)

