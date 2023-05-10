import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y)

#Question 2. Part A
train_points=int(len(X)*0.7)

train_X=pd.DataFrame(X[:train_points])	
train_y=pd.Series(y[:train_points],dtype="category")

test_X=pd.DataFrame(X[train_points:])
test_y=pd.Series(y[train_points:],dtype="category")	

for criteria in ['information_gain', 'gini_index']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(train_X, train_y)
    y_hat = tree.predict(test_X)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, test_y))
    for cls in pd.Series(y).unique():
        print('Precision: ', precision(y_hat, test_y, cls))
        print('Recall: ', recall(y_hat, test_y, cls))

#Question 2. Part B
outer_loops=5
inner_loop=4

tree_depths=[2,5]
best_depths=[]

for i in range(outer_loops):
    test_index= np.arange(i*(len(X)//outer_loops),(i+1)*(len(X)//outer_loops),1)
    train_indexes=np.delete(np.arange(0,100,1),test_index)
    all_fold_val_acc=[]
    for j in range(inner_loop):

        number=len(train_indexes)//inner_loop 
        val_set=train_indexes[j*number:(j+1)*number]
        train_set=np.setdiff1d(train_indexes,val_set)

        val_X_df=pd.DataFrame(X[val_set])
        val_Y=pd.Series(y[val_set],dtype="category")

        train_X_df=pd.DataFrame(X[train_set])
        train_Y=pd.Series(y[train_set],dtype="category")
        accuracy_arr=[]
        for k in range(len(tree_depths)):
            depth_curr_max=tree_depths[k]
            tree = DecisionTree(criterion='information_gain', max_depth = depth_curr_max)
            tree.fit(train_X_df, train_Y)

            y_hat = tree.predict(val_X_df) 
            accuracy_= accuracy(y_hat, val_Y)
            accuracy_arr.append(accuracy_)

        all_fold_val_acc.append(accuracy_arr)

    best_depth_arg=np.argmax(np.mean(all_fold_val_acc,axis=0))
    best_depths.append(tree_depths[best_depth_arg])
    
optimal_depth=pd.Series(best_depths).mode()[0]
print(optimal_depth)