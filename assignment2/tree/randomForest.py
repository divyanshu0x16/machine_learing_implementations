import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .base import DecisionTree
from .RFBase import RFBase

from scipy import stats
from joblib import Parallel, delayed

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.estimators = []
        self.unique_classes = None
        self.X = None
        self.y = None

        self.X_list=[]
        self.y_list=[]

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        self.unique_classes=len(y.unique())

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

        for i in range(self.n_estimators):
            tree = RFBase(criterion = self.criterion)
            tree.fit(X_list[i], y_list[i])
            self.estimators.append(tree)
            

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
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
        
    def __plotter(self,X, y, estimator, title, ax):
        X=np.array(X)
        y=np.array(y)

        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

        x1grid = np.arange(min1, max1, 0.05,dtype=np.float32)
        x2grid = np.arange(min2, max2, 0.05,dtype=np.float32)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = np.hstack((r1,r2))

        averages = np.tile( np.mean(X[:,2:], axis = 0), (grid.shape[0], 1))
        grid =  np.column_stack((grid, averages))

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
    
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        for estimator in self.estimators:
            estimator.plot()
        print("Estimators have been plotted")

        fig, ax = plt.subplots(nrows=1, ncols=self.n_estimators)
        for i in range(self.n_estimators):
            self.__plotter(self.X_list[i], self.y_list[i], self.estimators[i], 'Estimator: ' + str(i), ax[i])

        #For Figure - 2
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        self.__plotter(self.X, self.y, self, 'Random Forest Classifier', ax2)

        return fig,fig2



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        self.estimators = []
        self.unique_classes = None
        self.X = None
        self.y = None

        self.X_list=[]
        self.y_list=[]

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        self.unique_classes=len(y.unique())

        X_copy = self.X.copy()
        y_copy = self.y.copy()

        X_list = []
        y_list = []
        
        for i in range(self.n_estimators):
            sampled_X=X_copy.sample(n=math.floor(0.8 * len(y_copy)), replace=True)
            index_sampled=sampled_X.index

            y_sampled=y_copy[index_sampled].reset_index()
            sampled_X=sampled_X.reset_index(drop=True)
            y_sampled=pd.Series(np.array(y_sampled)[:,1])
            
            X_list.append(sampled_X)
            y_list.append(y_sampled)
        
        for i in range(self.n_estimators):
            tree = RFBase(criterion = self.criterion)
            tree.fit(X_list[i], y_list[i])
            self.estimators.append(tree)
        
        self.X_list=X_list
        self.y_list=y_list


    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
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

    def __plotter(self,X, y, estimator, title, ax):
        X=np.array(X)
        y=np.array(y)

        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

        x1grid = np.arange(min1, max1, 0.05,dtype=np.float32)
        x2grid = np.arange(min2, max2, 0.05,dtype=np.float32)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = np.hstack((r1,r2))

        averages = np.tile( np.mean(X[:,2:], axis = 0), (grid.shape[0], 1))
        grid =  np.column_stack((grid, averages))

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
    
    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        for estimator in self.estimators:
            estimator.plot()
        print("Estimators have been plotted")

        fig, ax = plt.subplots(nrows=1, ncols=self.n_estimators)
        for i in range(self.n_estimators):
            self.__plotter(self.X_list[i], self.y_list[i], self.estimators[i], 'Estimator: ' + str(i), ax[i])

        #For Figure - 2
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        self.__plotter(self.X, self.y, self, 'Random Forest Classifier', ax2)

        return fig,fig2
