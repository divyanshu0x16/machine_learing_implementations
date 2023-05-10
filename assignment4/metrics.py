from typing import Union
import pandas as pd

import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    y_true = y_hat == y
    return (y_true==True).sum()/len(y)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    y_hat_rel = y_hat[y_hat==cls]
    y_actual = y[y_hat_rel.index]
    y_precision = y_hat_rel == y_actual
    return (y_precision == True).sum()/len(y_hat_rel)


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    y_rel = y[y == cls]
    y_pred_rel = y_hat[y_rel.index]
    y_recall = y_pred_rel == y_rel
    return (y_recall == True).sum()/len(y_rel)
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    return (((y_hat - y)**2).mean())**0.5


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    return abs((y_hat - y)).mean()
