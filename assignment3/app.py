import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import streamlit as st

st.set_page_config(page_title='Taylor Series Approximations', layout="wide")
st.title('Taylor Series Approximations')

st.markdown('''A Taylor series is a mathematical representation of a function as an infinite sum of terms. Any non-polynomial function can be represented by a Taylor series as an infinite summation of polynomials and after a few terms, it can very close approximations to the original function.

The formula for the Taylor series is given by:''')

st.latex(r"f(x) = f(a) + \frac{f'(a)}{1!}(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots")

st.markdown('''where f(a) is the value of the function at x=a and f’(a), f’‘(a), f’‘’(a), etc., are the first, second, third, etc., derivatives of the function evaluated at x=a.

**In our representation, we have considered x = 0, for plotting the graphs. For plotting RMSE, we limited 'x' to range [-4, 4] since taylor series approximation which we are using are around x=0.**''')

x = sp.symbols('x')
sin_func = sp.sin(x)
cos_func = sp.cos(x)
ex_func = sp.exp(x)

with st.sidebar:
    function = st.selectbox('Select a function to approximate', ('sin(x)', 'cos(x)', 'e^x'))
    order = st.slider('Order of approximation', min_value=1, max_value=10, value=1)

def taylor_series(func, order):
    return sp.series(func, x, x0=0, n=order+1).removeO()

def app():
    x_range = np.linspace(-10, 10, 200)

    if function == 'sin(x)':
        taylor = taylor_series(sin_func, order)
        true_vals = np.sin(x_range)
        label = 'sin(x)'

    elif function == 'cos(x)':
        taylor = taylor_series(cos_func, order)
        true_vals = np.cos(x_range)
        label = 'cos(x)'

    elif function == 'e^x':
        taylor = taylor_series(ex_func, order)
        true_vals = np.exp(x_range)
        label = 'e^x'

    taylor_func = sp.lambdify(x, taylor, 'numpy')
    taylor_vals = taylor_func(x_range)
    
    #For cos(x) = 1
    if(type(taylor_vals)==int):
        taylor_vals=np.ones(200)

    rmse_arr = []
    for i in range(1,11):

        if(function == 'sin(x)'):
            rmse_func = sp.lambdify(x, taylor_series(sin_func, i), 'numpy')

        elif (function == 'cos(x)'):
            rmse_func = sp.lambdify(x, taylor_series(cos_func, i), 'numpy')

        elif (function == 'e^x'):
            rmse_func =  sp.lambdify(x, taylor_series(ex_func, i), 'numpy')
        
        curr_taylor_vals = rmse_func(x_range)
        if(type(curr_taylor_vals)==int):
            curr_taylor_vals=np.ones(200)

        rmse_arr.append(np.sqrt(np.mean(np.sum((curr_taylor_vals[60:140]-true_vals[60:140])**2)))) #Plotted only for values between -4 and 4

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.plot(x_range, true_vals, label=label)
    ax1.plot(x_range, taylor_vals, label=f'Taylor series approx. at x = 0(order {order})', linestyle='--')
    ax1.legend()
    ax1.set_xlim([-7, 7])
    ax1.set_ylim([-10, 10])
    ax1.grid(True)
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.axvline(x=0, color='black', linestyle='--')

    fig.subplots_adjust(wspace=0.25, hspace=None)

    ax2.plot(rmse_arr)
    ax2.set_xlabel('Order')
    ax2.set_ylabel('RMSE')
    ax2.grid(True)

    ax1.set_title('Function Plot')
    ax2.set_title('Plot for RMSE vs. Order')

    st.pyplot(fig)

app()