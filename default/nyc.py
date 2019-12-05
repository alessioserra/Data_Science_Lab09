'''
Created on 4 dic 2019

@author: zierp
'''
import numpy as np
from sklearn.model_selection._split import train_test_split
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import r2_score
import pandas as pd

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
X = X.dropna()
# Build y containing price
y = np.array(X.loc[:,['price']])
X = X.drop(columns=['id','name','host_name','latitude','longitude','price','last_review'])
""""""

# Encode categorical attributes
X = pd.get_dummies(X, columns=['host_id','neighbourhood_group','neighbourhood','room_type'], drop_first=True)

#Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42, shuffle=True)
reg = RandomForestRegressor(n_estimators=100)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print("Accuracy: ",r2_score(y_pred, y_test))

