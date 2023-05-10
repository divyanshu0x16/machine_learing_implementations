import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Read real-estate data set
# ...
# 
data = pd.read_fwf("auto-mpg.data")
data.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
#Drop Cateogorical Values
data = data.drop(['car name', 'cylinders', 'model year', 'origin'], axis = 1)
data = data.replace('?',np.nan)
data = data.dropna()
data.columns = np.arange(0,len(data.columns),1)
data = data.reset_index()

y = data[0]
X = data[data.columns[1:]]

folds=3
max_depths=[2,7,20]

for depth in max_depths:
    for i in range(folds):

        test_index= np.arange(i*(len(X)//folds),(i+1)*(len(X)//folds),1)
        train_indexes=np.delete(np.arange(0,len(X),1),test_index)
        
        X_copy=X.copy().astype(float)
        y_copy=y.copy().astype(float)

        train_X=pd.DataFrame(X_copy.iloc[train_indexes])
        train_y=pd.Series(y_copy.iloc[train_indexes])

        test_X=pd.DataFrame(X_copy.iloc[test_index])
        test_y=pd.Series(y_copy.iloc[test_index])
        
        tree = DecisionTree(criterion='information_gain', max_depth = depth)
        tree.fit(train_X, train_y)

        y_hat = tree.predict(test_X) 
        y_hat.index = test_y.index

        print('Current Fold: ', i,"Max_Depth :", depth)
        print('\tRMSE: ', rmse(y_hat, test_y))
        print('\tMAE: ', mae(y_hat, test_y))
        
        sklearn_reg= DecisionTreeRegressor(max_depth = depth)
        sklearn_reg.fit(train_X,train_y)
        y_hat_sk=sklearn_reg.predict(test_X)

        print('\tRMSE SKLEARN: ', rmse(y_hat_sk, test_y))
        print('\tMAE SKLEARN: ', mae(y_hat_sk, test_y))
        