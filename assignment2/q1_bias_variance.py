import numpy as np
import matplotlib.pyplot as plt

from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from math import floor

np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps

x, y= shuffle(x, y, random_state=42)
#for plotting
# plt.plot(x, y, 'o')
# plt.plot(x, x**2 + 1, 'r-')
#plt.show()

def bias(depth,features,labels):
    regressor = DecisionTreeRegressor(random_state=42,max_depth=depth)
    features=np.expand_dims(features,axis=1)
    regressor.fit(features,labels)
    preds=regressor.predict(features)
    mse=((preds-labels)**2).sum()
    return mse
    
depths=[3,4,5,6,7,8,9,10,None]
bias_regression=[]

for depth in depths:
    bias_regression.append(bias(depth,x,y))

plt.subplot(1, 2, 1)
plt.plot(bias_regression,color='green')
plt.xlabel("Depth")
plt.ylabel("Bias")

def variance(depth,features,labels):
    regressor = DecisionTreeRegressor(random_state=42,max_depth=depth)
    features=np.expand_dims(features,axis=1)
    regressor.fit(features,labels)
    preds=regressor.predict(features)
    return np.var(np.array(preds))

X_list=[]
y_list=[]
num_samples=50

training_index=np.arange(floor(0.7*len(y)))
testing_index=np.arange(len(y))[floor(0.7*len(y)):]
X_test=x[testing_index]
y_test=y[testing_index]

for i in range(num_samples):
    choice=np.random.choice(len(training_index),floor(0.5*len(y)))
    x_curr=x[choice]
    y_curr=y[choice]
    X_list.append(x_curr)
    y_list.append(y_curr)

variance_arr=[]
for depth in depths:
    preds=[]
    for i in range(len(X_list)):
        tree=DecisionTreeRegressor(random_state =42, max_depth=depth)
        tree.fit(np.expand_dims(X_list[i],axis=1),y_list[i])
        pred=tree.predict(np.expand_dims(X_test,axis=1))
        preds.append(pred)
    preds=np.array(preds)
    variance_mean=np.var(preds,axis=0).mean()
    variance_arr.append(variance_mean)
    
plt.subplot(1, 2, 2)
plt.plot(variance_arr,color='red')
plt.xlabel("Depth")
plt.ylabel("Variance")

plt.tight_layout()
plt.show()