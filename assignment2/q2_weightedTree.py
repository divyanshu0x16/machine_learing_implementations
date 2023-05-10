import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from typing import Union
from tree.base import DecisionTree
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from metrics import *

## claasfication dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X_orig, y_orig = X, y
weights = np.random.rand(len(y))
X, y, weights = shuffle(X, y, weights, random_state=42)

X = pd.DataFrame(X)
y = pd.Series(y, dtype = "category")
weights = pd.Series(weights)

train_indexes= np.arange(0,int(0.7*len(y)),1)
test_indexes=np.delete(np.arange(0,len(y),1),train_indexes)

train_X=pd.DataFrame(X.iloc[train_indexes])
train_y=pd.Series(y.iloc[train_indexes], dtype="category")
test_X=pd.DataFrame(X.iloc[test_indexes])
test_y=pd.Series(y.iloc[test_indexes], dtype="category")

train_weight=pd.DataFrame(weights.iloc[train_indexes])

## compare both the trees
tree = DecisionTree(criterion='information_gain', max_depth=3)
tree.fit(train_X, train_y, weights=train_weight) #Give empty series, in case of default 1 weights
y_hat = tree.predict(test_X)

sklearn_reg = DecisionTreeClassifier(max_depth=3,criterion='entropy')
sklearn_reg.fit(train_X, train_y, np.array(train_weight).squeeze())
sklearn_y_hat = sklearn_reg.predict(test_X)
sklearn_y_hat=pd.Series(sklearn_y_hat)

#tree.plot()
test_y.index=y_hat.index

print('Accuracy: ', accuracy(y_hat, test_y), ' SKLearn Accuracy: ', accuracy(sklearn_y_hat, test_y))
for cls in pd.Series(y).unique():
    print('Precision: ', precision(y_hat, test_y, cls), ' SKLearn Precision: ', precision(sklearn_y_hat, test_y, cls))
    print('Recall: ', recall(y_hat, test_y, cls), ' SKLearn Recall: ', recall(sklearn_y_hat, test_y, cls))

## For Plotting
min1, max1 = X_orig[:, 0].min()-1, X_orig[:, 0].max()+1
min2, max2 = X_orig[:, 1].min()-1, X_orig[:, 1].max()+1

x1grid = np.arange(min1, max1, 0.01)
x2grid = np.arange(min2, max2, 0.01)

xx, yy = np.meshgrid(x1grid, x2grid)

r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

grid = np.hstack((r1,r2))

grid=pd.DataFrame(grid)
y_hat_grid=tree.predict(grid)
y_hat_grid=np.array(y_hat_grid)

zz = y_hat_grid.reshape(xx.shape)
plt.contourf(xx, yy, zz, cmap = 'Paired')

for class_value in range(2):
    row_ix = np.where(y_orig == class_value)
    plt.scatter(X_orig[row_ix, 0], X_orig[row_ix, 1], cmap='Paired')

plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Weighted Tree')
plt.show()