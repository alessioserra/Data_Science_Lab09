'''
Created on 4 dic 2019

@author: zierp
'''
import csv
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import r2_score
from sklearn.preprocessing import LabelEncoder

#ex1
X = []
y = []
"""
 which attribute (or set of attributes) you think could drive the price per night the most?
 can you detect any irregularity in any attribute distribution?
 if your regression model will fit on numerical data only, how could you handle categorical
attributes?
"""
with open("development.csv", encoding="utf-8") as file:
    for row in csv.reader(file, delimiter=","):
            X.append( [row[2], row[4], row[5] ,row[8], row[10], row[11] ,row[14] ,row[15] ])
            y.append(row[9])
   
X.pop(0)
y.pop(0)  

#Regression
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=30, random_state=42, shuffle=True)
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Accuracy f3: ",r2_score(y_pred, y_test))

