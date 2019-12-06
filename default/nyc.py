'''
Created on 4 dic 2019

@author: zierp
'''
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import r2_score
import pandas as pd
np.random.seed(1234)
#ex1
X = []
"""
 which attribute (or set of attributes) you think could drive the price per night the most?
 can you detect any irregularity in any attribute distribution?
 if your regression model will fit on numerical data only, how could you handle categorical
attributes?
"""

# Load data
X = pd.read_csv('development.csv')
# Discard rows with price = 0
X = X[X.price > 0 ]

"""Drop all useless columns and rows containing NaN"""
X.fillna(0.0, inplace=True)
# Build y containing price
y = np.array(X.loc[:,['price']])
# convert price in log(price)
#for idx,el in enumerate(y):
    #y[idx] = np.log(el)
X = X.drop(columns=['id','host_id','name','host_name','neighbourhood_group','minimum_nights','price','number_of_reviews','last_review'])
""""""
# Encode categorical attributes
X = pd.get_dummies(X, columns=['room_type','neighbourhood'], drop_first=True)
print(X.columns)
#Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=False)
reg = RandomForestRegressor(n_estimators=100, n_jobs=-1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

"""
# re-convert in number
for idx,el in enumerate(y_pred):
    y_pred[idx] = np.exp(el)
    
for idx,el in enumerate(y_test):
    y_test[idx] = np.exp(el)
"""
print("Accuracy: ",r2_score(y_pred, y_test))

# Evaluation
X_eval = pd.read_csv('evaluation.csv')
