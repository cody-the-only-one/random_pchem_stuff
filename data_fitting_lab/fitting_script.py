import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re # Used for flexible data loading

# --- 1. DEFINE YOUR MODEL FUNCTION ---
# The first argument must be the independent variable (x),
# followed by all the parameters you want the fit to find.
# TODO: You will change this function for each experiment.
def fit_function(x, a, b, c):
    """A model function: f(x) = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

# --- 2. CONFIGURE THE SCRIPT ---
# TODO: Change these two lines for each new data set.
data_file = '/content/sample_data/my_exp_data.txt'
num_columns = 2

# --- 3. LOAD THE DATA ---
try:
    with open(data_file, 'r') as f:
        file_content = f.read()
    # Split the content by any mix of spaces, commas, or tabs
    data = [float(num) for num in re.split(r'[ ,\t\n]+', file_content.strip()) if num]
    data_array = np.array(data).reshape(-1, num_columns)
    
    # Generic unpacking of columns based on num_columns
    columns = [data_array[:, i] for i in range(num_columns)]
    print(f"Loaded {len(columns[0])} data points from {data_file}.")
except Exception as e:
    print(f"Error processing {data_file}: {e}")
    print("Ensure the file path is correct and it contains the right number of columns.")
    exit()

# --- 4. PROCESS DATA AND SET INITIAL GUESSES ---
# TODO: Assign columns to x_data and y_data. Perform any needed math.
x_data = columns[0]
y_data = columns[1]

# TODO: Provide reasonable initial guesses for your fitting parameters.
# The order must match the parameters in your fit_function definition!
initial_guesses = [1.0, -1.0, 1.0]

# --- 5. PERFORM THE CURVE FIT ---
try:
    # The main fitting command from SciPy
    popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses)

    # --- 6. PRINT AND VISUALIZE THE RESULTS ---
    print("\n--- Fit Results ---")
    param_names = fit_function.__code__.co_varnames[1:len(popt)+1]
    for i, name in enumerate(param_names):
        print(f"Optimal {name} = {popt[i]:.4f}")
    
    print("\n--- Parameter Errors ---")
    perr = np.sqrt(np.diag(pcov)) # Standard errors
    for i, name in enumerate(param_names):
        print(f"Standard error for {name} = {perr[i]:.4f}")

    x_fit_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit_line = fit_function(x_fit_line, *popt)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Experimental Data', color='blue')
    plt.plot(x_fit_line, y_fit_line, 'r-', label='Fitted Function')
    plt.title('Fit of Model to Experimental Data') # TODO: Change title
    plt.xlabel('x-axis label (units)') # TODO: Change axis labels
    plt.ylabel('y-axis label (units)') # TODO: Change axis labels
    plt.legend()
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"\nCurve fitting failed: {e}")
    print("This often means your initial guesses are too far off. Try adjusting them.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
