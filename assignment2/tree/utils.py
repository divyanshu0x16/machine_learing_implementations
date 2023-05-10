import math
import pandas as pd
import numpy as np
np.seterr(divide='ignore')

def entropy(Y: pd.Series , weights: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    total_weight = weights.sum()
    entropy = 0

    for value in Y.unique():
        attribute_subset = Y[Y == value]
        weight_subset = weights[attribute_subset.index]
        weighted_prob = weight_subset.sum() / total_weight
        class_entropy = math.log(weighted_prob, 2) * weighted_prob * (-1)
        entropy += class_entropy
        
    return entropy

def gini_index(Y: pd.Series , weights: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    gini=0
    weight_total=weights.sum()
    for val in Y.unique():
        weight_subset=weights[Y==val]
        weight_subset_sum=weight_subset.sum()
        weight_normalised=weight_subset_sum/weight_total
        gini+=weight_normalised*(1-weight_normalised)
    return gini


def information_gain(Y: pd.Series, attr: pd.Series , weights: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    weighted_entropy = 0
    total_weight = weights.sum()

    for val in attr.unique():
        y_subset = Y[attr==val]

        weight_subset = weights[attr==val]
        weight_subset_sum = weight_subset.sum()

        entropy_= entropy(y_subset,weight_subset)
        weighted_entropy += entropy_*(weight_subset_sum/total_weight)

    return entropy(Y, weights) - weighted_entropy

def var_red(Y: pd.Series, attr: pd.Series, weights: pd.Series) -> float:
    """
    Function to calculate reduction in variance for Discrete Input, Real Output
    """
    total_var = Y.var()
    weighted_variance = 0
    for val in attr.unique():
        vals = Y[attr==val]
        weighted_variance += vals.var(ddof = 0)*(len(vals)/len(Y))

    return total_var-weighted_variance

def riro_loss(Y: pd.Series, attr: pd.Series , weights: pd.Series) -> float:
    loss = 0
    for val in attr.unique():
        vals = Y[attr==val]
        mean = vals.mean()
        diff = vals-mean
        loss += (diff**2).sum()
    return -1*loss


# data = np.array([0, 0, 0, 1, 1, 1])
# weights=np.array([0.1, 0.1, 0.3, 0.1, 0.1, 0.3])
# ser = pd.Series(data)
# weights=pd.Series(weights)
# x=ser.value_counts(normalize=True)


# df = pd.DataFrame({
#     'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
#     'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
#     'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# })
# weights=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4]
# print(df)
# print(var_red(df['PlayTennis'], df['Wind']))
# print(gini_index(df['PlayTennis'], pd.Series(weights)))
# y=pd.Series(np.array([1,2,3,4,5]))
# attr=pd.Series(np.array([0,0,1,1,1]))
# print(riro_loss(y, attr))

