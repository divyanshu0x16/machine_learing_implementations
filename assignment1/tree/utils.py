import pandas as pd
import numpy as np
np.seterr(divide='ignore')

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    x= Y.value_counts(normalize=True)
    return -1*(x*np.log2(x)).sum()


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    x= Y.value_counts(normalize=True)
    return (x*(1-x)).sum()


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    set_sizes=pd.crosstab(attr,Y).sum(axis=1)
    df_norm=pd.crosstab(attr,Y,normalize='index')

    df_entropy=-1*(df_norm*np.log2(df_norm)).fillna(0)
    entropies = df_entropy.sum(axis=1)

    return entropy(Y) - (entropies*(set_sizes/len(Y))).sum(axis=0)

def var_red(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate reduction in variance for Discrete Input, Real Output
    """
    total_var = Y.var()
    weighted_variance = 0
    for val in attr.unique():
        vals = Y[attr==val]
        weighted_variance += vals.var(ddof = 0)*(len(vals)/len(Y))

    return total_var-weighted_variance

def riro_loss(Y: pd.Series, attr: pd.Series) -> float:
    loss = 0
    for val in attr.unique():
        vals = Y[attr==val]
        mean = vals.mean()
        diff = vals-mean
        loss += (diff**2).sum()
    return -1*loss


# data = np.array(['g', 'e', 'e', 'k', 's'])
 
# ser = pd.Series(data)
# x=ser.value_counts(normalize=True)

# print(entropy(ser))

# df = pd.DataFrame({
#     'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
#     'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
#     'PlayTennis': [20, 24, 40, 50, 60, 10, 4, 10, 60, 40, 45, 40, 35, 20]
# })
# print(df)
# print(var_red(df['PlayTennis'], df['Wind']))

# y=pd.Series(np.array([1,2,3,4,5]))
# attr=pd.Series(np.array([0,0,1,1,1]))
# print(riro_loss(y, attr))

