# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:57:06 2019

@author: Sayantan Ghosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Water_Survey.csv')
X = dataset.iloc[:,0:1].values
Y = dataset.iloc[:,1:2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

  #Fitting regressor to dataset
 
  from sklearn.svm import SVR
  regressor = SVR(kernel = 'rbf')
  regressor.fit(X,Y)
 
  
  Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))#Getting the original Value of the Scaled Value
  plt.scatter(X,Y)
  plt.plot(X,regressor.predict(X), color = 'blue')
  plt.title("Thruth and Bluff")
  plt.xlabel("Level")
  plt.ylabel("Slary")
  plt.show()