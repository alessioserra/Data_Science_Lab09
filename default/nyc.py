'''
Created on 4 dic 2019
@author: zierp
'''
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing.data import StandardScaler
from sklearn.model_selection import GridSearchCV

def make_submission(prediction, sub_name):
    my_submission = pd.DataFrame({'Id':pd.read_csv('evaluation.csv').id,'Predicted':prediction})
    my_submission.to_csv('{}.csv'.format(sub_name),index=False)
    print('A submission file has been made')

np.random.seed(1234)
#ex1
X = []
"""
 which attribute (or set of attributes) you think could drive the price per night the most?
 can you detect any irregularity in any attribute distribution?
 if your regression model will fit on numerical data only, how could you handle categorical
attributes?
"""

# Load data and remove neighbourhood from X that are missing in X_eval
X = pd.read_csv('development.csv')
X_eval = pd.read_csv('evaluation.csv')
ng_dev = set(X['neighbourhood'])
ng_eval = set(X_eval['neighbourhood'])
remove = []
for el in ng_dev:
    if not el in ng_eval:
        remove.append(el)
for el in remove:
    X = X[X.neighbourhood != el ]
# Discard rows with price = 0
X = X[X.price > 0 ]

"""Drop all useless columns and rows containing NaN"""
X.fillna(0.0, inplace=True)

# Build y containing price
#X['price'] = X['price'].map(lambda price: np.log(price)) #log scale sui price
y = np.array(X.loc[:,['price']])
# convert price in log(price)
#for idx,el in enumerate(y):
    #y[idx] = np.log(el)
X = X.drop(columns=['name','host_name','neighbourhood_group','minimum_nights','price','number_of_reviews','last_review'])
""""""

# Encode X
X = pd.get_dummies(X, columns=['room_type','neighbourhood'], drop_first=True)
# Encode X_eval
X_eval.fillna(0.0, inplace=True)
X_eval = X_eval.drop(columns=['name','host_name','neighbourhood_group','minimum_nights','number_of_reviews','last_review'])
X_eval = pd.get_dummies(X_eval, columns=['room_type','neighbourhood'], drop_first=True)

"""Scaling
x_scaler = StandardScaler()
x_scaler.fit(X)
X = x_scaler.transform(X)
X_eval = x_scaler.transform(X_eval)
"""

# Regression with XGBRegressor
reg = XGBRegressor(max_depth=5, min_child_weight=1, n_estimators = 105, n_jobs=-1)
#param_grid = {'n_estimators' : [90,100,110]} 
#gridsearch = GridSearchCV(reg, param_grid, scoring='r2', cv=5)
#gridsearch.fit(X,y)
#print(gridsearch.best_params_['n_estimators'])

reg.fit(X,y)

# Predict
XGBpredictions = reg.predict(X_eval)
#XGBpredictions = [float(np.exp(el)) for el in XGBpredictions ] rescaling price

# Make submission file
make_submission(XGBpredictions,'result')