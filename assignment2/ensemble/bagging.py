import math
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100, n_jobs=1):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.estimators=[]
        self.unique_classes=None
        self.X = None
        self.y = None
        self.n_jobs = n_jobs
        self.X_list=[]
        self.y_list=[]

    def __plotter(self,X, y, estimator, title, ax):
        X=np.array(X)
        y=np.array(y)

        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = np.hstack((r1,r2))

        grid=pd.DataFrame(grid)
        y_hat_grid=estimator.predict(grid)
        y_hat_grid=np.array(y_hat_grid)

        zz = y_hat_grid.reshape(xx.shape)
        ax.contourf(xx, yy, zz, cmap = 'Paired')

        for class_value in range(self.unique_classes):
            row_ix = np.where(y == class_value)
            ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
        ax.set_title(title)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')

    def __fit_helper(self, X, y):
        model_ = copy.deepcopy(self.base_estimator)
        model_.fit(X,y)
        self.estimators.append(model_)

        return model_

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X=X
        self.y=y
        self.unique_classes=len(y.unique())

        def replicate(X, y):
            return self.__fit_helper(X, y)

        X_copy = self.X.copy()
        y_copy = self.y.copy()

        X_list = []
        y_list = []
        
        for i in range(self.n_estimators):
            sampled_X=X_copy.sample(n=math.floor(0.8 * len(y_copy)), replace=True)
            index_sampled=sampled_X.index

            y_sampled=y_copy[index_sampled].reset_index()
            sampled_X=sampled_X.reset_index(drop=True)
            y_sampled=pd.Series(np.array(y_sampled)[:,1], dtype="category")
            
            X_list.append(sampled_X)
            y_list.append(y_sampled)
        self.X_list=X_list
        self.y_list=y_list
        
        start_time = time.perf_counter()
        estimators = Parallel(n_jobs=1)(delayed(replicate)(X_list[i], y_list[i]) for i in range(self.n_estimators))
        finish_time = time.perf_counter()
        print(f"Normal impl. finished in {finish_time-start_time} seconds")

        start_time = time.perf_counter()
        estimators = Parallel(n_jobs=self.n_jobs)(delayed(replicate)(X_list[i], y_list[i]) for i in range(self.n_estimators))
        finish_time = time.perf_counter()
        print(f"Parallel impl. finished in {finish_time-start_time} seconds")

        self.estimators = estimators

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred_array=[]
        for estimator in self.estimators:
            pred=estimator.predict(X)
            pred_array.append(np.array(pred))
            
        pred_array=np.array(pred_array)
        mode_array=stats.mode(pred_array,axis=0)[0][0]
        return pd.Series(mode_array)

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        #For Figure - 1
        fig, ax = plt.subplots(nrows=1, ncols=self.n_estimators)
        for i in range(self.n_estimators):
            self.__plotter(self.X_list[i], self.y_list[i], self.estimators[i], 'Estimator: ' + str(i), ax[i])
        #For Figure - 2
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        self.__plotter(self.X, self.y, self, 'Bagging', ax2)
        
        return fig, fig2

