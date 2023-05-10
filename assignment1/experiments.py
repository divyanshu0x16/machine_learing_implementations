import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from tqdm import tqdm
np.random.seed(42)
num_average_time = 10

# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def calculate_time(X,y,depth):

    fit_time_arr=[]
    predict_time_arr=[]
    
    for i in tqdm(range(num_average_time)):
        tree=DecisionTree(criterion='information_gain',max_depth=depth)
        t1=time.time()
        tree.fit(X,y)
        t2=time.time()
        fit_time_arr.append(t2-t1)

        t1=time.time()
        tree.predict(X)
        t2=time.time()
        predict_time_arr.append(t2-t1)

    return np.mean(fit_time_arr), np.std(fit_time_arr), np.mean(predict_time_arr), np.std(predict_time_arr)

# Function to plot the results
def plot_results(results, title):

    fig, ax = plt.subplots()

    keys = list(results.keys())
    values = np.array([results[key] for key in keys])

    x = np.arange(len(keys))

    ax.bar(x, values[:,0], label='N = 5, M = 5', bottom=0, color='red')
    ax.bar(x, values[:,1], label='N = 20, M = 20', bottom=values[:,0], color='blue')
    ax.bar(x, values[:,2], label='N = 100, M = 100', bottom=values[:,0]+values[:,1], color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Time(s)')
    ax.legend()

    plt.title(title,fontsize=20)
    plt.show()
    
# Function to create fake data (take inspiration from usage.py)
def generate_data( _case, n, m):

    if(_case == 'dido'):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(m)})
        y = pd.Series(np.random.randint(2, size = n), dtype="category")
    elif(_case == 'diro'):
        X = pd.DataFrame({i:pd.Series(np.random.randint(2, size = n), dtype="category") for i in range(m)})
        y = pd.Series(np.random.randn(n))
    elif(_case == 'rido'):
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randint(2, size = n), dtype="category")
    else:
        X = pd.DataFrame(np.random.randn(n, m))
        y = pd.Series(np.random.randn(n))

    return X, y

def generate_timings(case):
    
    X_5,y_5=generate_data(case, 5, 5)
    X_20,y_20=generate_data(case, 20, 20)
    X_100,y_100=generate_data(case, 100, 100)

    depths = [2, 7, 20]

    fit_results = {2: [], 7: [], 20: []}
    predict_results = {2: [], 7: [], 20: []}

    for depth in tqdm(depths, desc='depth', total=len(depths)):
        mean_fit_5,mean_std_5,mean_predict_5,mean_predict_5=calculate_time(X_5,y_5,depth=depth)
        mean_fit_20,mean_std_20,mean_predict_20,mean_predict_20=calculate_time(X_20,y_20,depth=depth)
        mean_fit_100,mean_std_100,mean_predict_100,mean_predict_100=calculate_time(X_100,y_100,depth=depth)

        fit_results[depth].append(mean_fit_5)
        fit_results[depth].append(mean_fit_20)
        fit_results[depth].append(mean_fit_100)
        
        predict_results[depth].append(mean_predict_5)
        predict_results[depth].append(mean_predict_20)
        predict_results[depth].append(mean_predict_100)

    return fit_results, predict_results

dido_fit, dido_predict = generate_timings('dido')
plot_results(dido_fit, 'DIDO Fit')
plot_results(dido_predict, 'DIDO Predict')
print(dido_fit, dido_predict)

rido_fit, rido_predict = generate_timings('rido')
plot_results(rido_fit, 'RIDO Fit')
plot_results(rido_predict, 'RIDO Predict')
print(rido_fit, rido_predict)

diro_fit, diro_predict = generate_timings('diro')
plot_results(diro_fit, 'DIRO Fit')
plot_results(diro_predict, 'DIRO Predict')
print(diro_fit, diro_predict)

riro_fit, riro_predict = generate_timings('riro')
plot_results(riro_fit, 'RIRO Fit')
plot_results(riro_predict, 'RIRO Predict')
print(riro_fit, riro_predict)






