import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        if(base_estimator == None):
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
        else:
            self.base_estimator = base_estimator
        self.n_estimators=n_estimators
        self.estimators=[]
        self.estimator_weights = []
        self.unique_classes=None
        self.X = None
        self.y = None
        self.all_weights = []

    def __plotter(self,X, y, weights, estimator, title, ax):
        X=np.array(X)
        y=np.array(y)
        weights=np.array(weights)

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
            ax.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired', s = weights[row_ix]*2500)
        ax.set_title(title)
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')

        
    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        weights=pd.Series(np.ones(len(y))/len(y))
        self.unique_classes=len(y.unique())
        
        for i in range(self.n_estimators):
            
            tree_ = copy.deepcopy(self.base_estimator)
            self.all_weights.append(np.array(weights))
            tree_.fit(X,y, weights)
            y_hat=tree_.predict(X)

            wrong_preds_bool=(y_hat!=y).astype(int)
            weighted_error=((wrong_preds_bool*weights).sum())/(weights.sum())

            if(weighted_error==0 or weighted_error >= 0.5):
                break
            alpha_m=0.5*np.log((1-weighted_error)/weighted_error)

            for i in range(len(weights)):
                if(pd.Series(y_hat).iloc[i]!=y.iloc[i]):
                    weights.iloc[i]*=np.exp(alpha_m)
                else:
                    weights.iloc[i]*=np.exp(-alpha_m)

            weights=weights/weights.sum()
            self.estimators.append(tree_)
            self.estimator_weights.append(alpha_m)
           
    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred_array=[]
        one_hot_encoding=np.zeros((len(X),self.unique_classes))

        for i in range(len(self.estimators)):
            estimator=self.estimators[i]
            weight=self.estimator_weights[i]
            prediction=estimator.predict(X)
        
            for j in range(len(prediction)):
                one_hot_encoding[j][int(pd.Series(prediction).iloc[j])]+=weight
            
        for i in one_hot_encoding:
            pred_array.append(np.argmax(i))

        return pd.Series(pred_array)

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        
        #For Figure - 1
        fig, ax = plt.subplots(nrows=1, ncols=self.n_estimators)
        for i in range(self.n_estimators):
            self.__plotter(self.X, self.y, self.all_weights[i], self.estimators[i], 'Alpha: ' + str(self.estimator_weights[i]), ax[i])
        # plt.show()
        
        #For Figure - 2
        fig2, ax2 = plt.subplots(nrows=1, ncols=1)
        self.__plotter(self.X, self.y, np.ones(len(self.all_weights[0]))*0.02, self, 'AdaBoost', ax2)
        # plt.show()

        return fig, fig2
