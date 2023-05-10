# Taylor Series Approximations

This is a Python script that approximates the sine, cosine, and exponential functions using Taylor series. The script uses the SymPy library to compute the Taylor series and the NumPy and Matplotlib libraries to plot the results.

The script is built using the Streamlit library which allows the user to select the function to approximate and the order of the approximation. The script then computes the Taylor series of the selected function up to the specified order, and plots the result alongside the true values of the function.

## Requirements

- Python 3.x
- Streamlit
- NumPy
- SymPy
- Matplotlib

## Usage

To run the script, execute the following command in a terminal:

```bash
streamlit run app.py
```

Once the script is running, select the function to approximate and the order of the approximation using the sidebar, and the script will plot the result.

## Live Demo

```
https://divyanshu0x16-taylor-series-approximations-app-1fao0k.streamlit.app/
```

You can find a live demo of this script [here](https://divyanshu0x16-taylor-series-approximations-app-1fao0k.streamlit.app/). Simply select the function and the order of approximation using the sidebar, and the plot will update in real-time.

## Example

![exmample_plot](https://user-images.githubusercontent.com/62815174/234184164-02780c81-ffd7-40b2-8226-29da2a915cd5.png)

**For plotting RMSE, we limited 'x' to range [-4, 4] since Taylor Series Approximation which we are using are around x=0.**

In this example, the script approximates the sine function up to order 10. The true values of the sine function are shown in blue, and the Taylor series approximation is shown in orange. The right-hand plot shows the root mean square error (RMSE) of the approximation for orders 1 to 10. The RMSE decreases as the order of the approximation increases, as expected. 
