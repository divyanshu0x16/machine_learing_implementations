import copy
import sklearn
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

class GradientBoostedRegressor:
    def __init__(
        self, base_estimator, n_estimators=3, learning_rate=0.1
    ):  # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.first_val=None
        self.estimators=[]


    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        average_prediction=np.mean(y)
        self.first_val=average_prediction
        residuals=y-average_prediction
        for i in range(self.n_estimators-1):
            tree=copy.deepcopy(self.base_estimator)
            tree.fit(X,residuals)
            self.estimators.append(tree)
            residuals=residuals-self.learning_rate*tree.predict(X)
            if(np.sum(np.abs(residuals))==0):
                break

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred=np.ones(len(X))*self.first_val
        for i in range(len(self.estimators)):
            pred+=(self.learning_rate)*(self.estimators[i].predict(X))
        return pred
