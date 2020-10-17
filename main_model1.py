# -*- coding: utf-8 -*-
"""
Created on Fri May 15 00:28:17 2020

@author: USER
"""
import copy, os, math
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

"""
Sensors
'ACC', 'ECG', 'EMG', 'EDA', 'BVP', 'Temp', 'Resp'
"""


#clf1 = LogisticRegression(random_state=0)
#clf2 = KNeighborsClassifier( n_neighbors=10)
#clf3 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
#clf4 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
#clf5 = BaggingClassifier(clf3, max_samples=0.5, max_features=0.5)

if __name__ =="__main__":
      data = pd.read_csv("data/m14_merged.csv");
    
