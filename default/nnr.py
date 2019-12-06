from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
import string as str

def get_data():
    #get train data
    train_data_path ='development.csv'
    train = pd.read_csv(train_data_path)
    
    #get test data
    test_data_path ='evaluation.csv'
    test = pd.read_csv(test_data_path)
    
    return train, test

def get_combined_data():
    #reading train data
    train , test = get_data()

    target = train.price
    train.drop(['price'],axis = 1 , inplace = True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['id','host_id','name','host_name','neighbourhood_group','minimum_nights','number_of_reviews','last_review'], inplace=True, axis=1)
    return combined, target

'''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
def get_cols_with_no_nans(df,col_type):
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans

def oneHotEncode(df,colNames):
    for col in colNames:
        if( df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)

            #drop the encoded column
            df.drop([col],axis = 1 , inplace=True)
    return df

#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()

# Get the columns that do not have any missing values 
num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')

# Lets see how many columns we got
print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

# Encoding
combined = oneHotEncode(combined, cat_cols)

# The correlation between the features
#train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target
C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))
sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()

# Load data
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

# Initialize neural network
NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

#Checkpoint
checkpoint_name = 'best.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

"""Setup NN"""
NN_model.fit(X, y, epochs=10, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# Load wights file of the best model :
wights_file = checkpoint.filepath # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


def make_submission(prediction, sub_name):
    my_submission = pd.DataFrame({'Id':pd.read_csv('evaluation.csv').id,'Predicted':prediction})
    my_submission.to_csv('{}.csv'.format(sub_name),index=False)
    print('A submission file has been made')
    
X_eval.fillna(0.0, inplace=True)
X_eval = X_eval.drop(columns=['id','host_id','name','host_name','neighbourhood_group','minimum_nights','number_of_reviews','last_review'])
X_eval = pd.get_dummies(X_eval, columns=['room_type','neighbourhood'], drop_first=True)

predictions = NN_model.predict(X_eval)
make_submission(predictions[:,0],'result')