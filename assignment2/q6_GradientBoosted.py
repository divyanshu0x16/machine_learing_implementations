import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

from metrics import *

from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import DecisionTree

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################

from sklearn.datasets import make_regression

X, y= make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)

# For plotting
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], y)

tree = DecisionTreeRegressor(max_depth=3)
gradBoostRegressor = GradientBoostedRegressor(base_estimator=tree,n_estimators=250,learning_rate=0.1)
gradBoostRegressor.fit(X, y)

y_hat = gradBoostRegressor.predict(X)
print('RMSE: ', rmse(pd.Series(y_hat), pd.Series(y)))

print('MAE: ', mae(pd.Series(y_hat), pd.Series(y)))