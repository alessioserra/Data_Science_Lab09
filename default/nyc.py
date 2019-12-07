'''
Created on 4 dic 2019

@author: zierp
'''
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor

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
y = np.array(X.loc[:,['price']])
# convert price in log(price)
#for idx,el in enumerate(y):
    #y[idx] = np.log(el)
X = X.drop(columns=['id','host_id','name','host_name','neighbourhood_group','minimum_nights','price','number_of_reviews','last_review'])
""""""
# Encode categorical attributes
X = pd.get_dummies(X, columns=['room_type','neighbourhood'], drop_first=True)

# Regression with XGBRegressor
reg = XGBRegressor(max_depth=5, min_child_weight=5)
reg.fit(X,y)

# Encode X_eval
X_eval.fillna(0.0, inplace=True)
X_eval = X_eval.drop(columns=['id','host_id','name','host_name','neighbourhood_group','minimum_nights','number_of_reviews','last_review'])
X_eval = pd.get_dummies(X_eval, columns=['room_type','neighbourhood'], drop_first=True)

# Predict
XGBpredictions = reg.predict(X_eval)

# Make submission file
make_submission(XGBpredictions,'result')
