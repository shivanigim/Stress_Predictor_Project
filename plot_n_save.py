# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:41:54 2020

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
#clf1 = LogisticRegression(random_state=0)
#clf2 = KNeighborsClassifier( n_neighbors=10)
#clf3 = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
#clf4 = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
#clf5 = BaggingClassifier(clf3, max_samples=0.5,max_features=0.5)

import os
import matplotlib as mt

if __name__ =="__main__":
    print ("God is great!")    

    files = os.listdir("test_cases_temp/")
    
    for item in files:
        data_temp = pd.read_csv("test_cases_temp/"+item)
        cols= data_temp.columns.tolist()

        for col in cols:
            data_temp_col = data_temp[col]
            
            #plot and save data data_temp_col, file name as below
            #File name convention
            print ("Filename:", item.split(".")[0]+"_"+col+".jpg")



