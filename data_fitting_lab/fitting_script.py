import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
# --- Data and Plotting ---
data_file = '/content/sample_data/test2'
num_columns = 2
xaxis_label = 'X-axis Label'
yaxis_label = 'Y-axis Label'
plot_title = 'Function Fit'

# --- Fitting Function Definition ---
# Define the equation as a string. Use 'x' for the independent variable.
# Ensure you use 'np.' for numpy functions like exp, sin, cos, etc.
fit_equation_string = "a * np.exp(b * x) + c"

# List the parameter names used in the equation string IN ORDER.
param_names = ['a', 'b', 'c']

# Provide initial guesses for each parameter, in the same order as param_names.
initial_guesses = [10.0, -1.0, 42.0]

# --- Sanity Check (ensures lists have the same length) ---
if len(param_names) != len(initial_guesses):
    raise ValueError("The number of 'param_names' must match the number of 'initial_guesses'.")

# ==============================================================================
# 2. DYNAMIC FUNCTION CREATION
# ==============================================================================

def create_fit_function(expression_string, parameter_names):
    """
    Creates a Python function from a mathematical string expression.

    Args:
        expression_string (str): The mathematical formula (e.g., "a * np.exp(b * x) + c").
        parameter_names (list): A list of strings with parameter names (e.g., ['a', 'b', 'c']).

    Returns:
        A callable function suitable for use with scipy.curve_fit.
    """
    # This is a factory: a function that returns another function.
    # The returned function will be our model for curve_fit.
    def fit_function(x, *params):
        # Create a dictionary to hold the value of each parameter
        # e.g., {'a': params[0], 'b': params[1], 'c': params[2]}
        param_dict = {name: value for name, value in zip(parameter_names, params)}

        # The 'locals' for the eval function. It needs to know 'x' and the parameter values.
        local_vars = {'x': x, **param_dict}
        
        # The 'globals' for the eval function. It needs to know 'np' for numpy functions.
        global_vars = {'np': np}

        # Evaluate the expression string within the given context
        return eval(expression_string, global_vars, local_vars)

    return fit_function

# Create the actual function object we will use for fitting
fit_function = create_fit_function(fit_equation_string, param_names)


# ==============================================================================
# 3. DATA LOADING
# ==============================================================================
try:
    with open(data_file, 'r') as f:
        file_content = f.read()
    data = [float(num) for num in re.split(r'[ ,\t\n]+', file_content.strip()) if num]
    data_array = np.array(data).reshape(-1, num_columns)
    
    column = []
    column.append(data_array[:, 0]) 
    for column_number in range(num_columns): # index column array starting at 1
        column.append(data_array[:, column_number]) 

    print(f"Loaded data with {len(data_array[:,0])} data points.")

except (IOError, ValueError, IndexError) as e:
    print(f"Error processing '{data_file}': {e}")
    print("Please check file path, num_columns, and data format.")
    exit()

# note: Python is typically uses 0-based indexing
# but here the index is the column number.
x_data = column[1]
y_data = column[2]#+column[3]

# ==============================================================================
# 4. PERFORM CURVE FIT AND DISPLAY RESULTS
# ==============================================================================
try:
    # `popt` will be an array of the optimal parameter values
    # `pcov` is the covariance matrix
    popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses)

    # --- Print and interpret the results dynamically ---
    print("\nFitting complete!")
    print("Optimal parameters:")
    perr = np.sqrt(np.diag(pcov)) # Standard errors on the parameters

    # Dynamically build the result string
    result_strings = []
    for i in range(len(popt)):
        name = param_names[i]
        value = popt[i]
        error = perr[i]
        print(f"  {name} = {value:.4f} +/- {error:.4f}")
        result_strings.append(f"{name}={value:.2f}")

    # --- Visualize the fit ---
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
    
    # Use the star (*) operator to unpack the popt array into individual arguments
    # This is equivalent to calling: fit_function(x_fit_line, popt[0], popt[1], popt[2], ...)
    y_fit_line = fit_function(x_fit_line, *popt)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Original Data', color='blue', s=20)
    
    # Create a dynamic label for the plot
    fit_label = f"Fit: {', '.join(result_strings)}"
    plt.plot(x_fit_line, y_fit_line, 'r-', label=fit_label)
    
    plt.title(plot_title)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.legend()
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"\nCurve fitting failed: {e}")
    print("This often means the fit could not converge.")
    print("Try adjusting the `initial_guesses` or checking your data and equation.")

    # --- Visualize the data with the INITIAL GUESSES to help debug ---
    print("\nPlotting data with initial guess parameters...")
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 100)
    
    # Unpack the initial_guesses list
    y_initial_line = fit_function(x_fit_line, *initial_guesses)
    
    initial_guess_label = f"Initial Guess: {', '.join([f'{n}={v:.2f}' for n, v in zip(param_names, initial_guesses)])}"

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Original Data', color='blue')
    plt.plot(x_fit_line, y_initial_line, 'g--', label=initial_guess_label) # Green dashed line for guess
    plt.title('FAILED FIT - Initial Guess Visualization')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")