import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import re # Import the regular expression module
# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
# This is the only section you should need to edit for a new analysis.

# --- Data and Plotting ---
data_file = '/content/lab_data/test1'
xaxis_label = 'X-axis Label'
yaxis_label = 'Y-axis Label'
plot_title = 'Function Fit'

# --- Fitting Function Definition ---
# Define the equation as a string. Use 'x' for the independent variable.
fit_equation_string = "a * np.exp(b * x) + c"
param_names = ['a', 'b', 'c']
initial_guesses = [-10.0, 1.0, 42.0]

# --- Data Processing Rules ---
def process_data_rules(column):
    # --- Define x and y data ---
    # `column(1)` is the first column, `column(2)` is the second, etc.
    x_data = column(1) 
    y_data = column(2)
    # y_data = column(2)**2/237.238 # or do math
    
    # --- Define y_errors to enable a weighted fit (optional) ---
    #y_errors = None              # Leave as `None` for a standard, unweighted fit.
    # y_errors = column(1)       # Use the 3rd column from your file for errors
    y_errors = 0.02             # Use a constant error for all points
    # y_errors = np.sqrt(y_data) # Calculate Poisson error from y-data
    
    return x_data, y_data, y_errors

# ---- Plotting Style variables ----
plot_title_fontsize = 18
axis_label_fontsize = 14
legend_fontsize = 12
tick_label_fontsize = 12
reporting_precision = 2 

# ==============================================================================
#           END OF USER CONFIGURATION - NO NEED TO EDIT BELOW THIS LINE
# ==============================================================================
def format_value(value, precision):
# Formats a number for display, using scientific notation for very large or very small numbers.
    if value == 0:
        return f"{0.0:.{precision}f}"
    abs_val = abs(value)
    if abs_val > 1e5 or abs_val < 0.1: # Use scientific
        return f"{value:.{precision}e}"
    else:  # Otherwise, use standard float formatting
        return f"{value:.{precision}f}"

# --- Sanity Check ---
if len(param_names) != len(initial_guesses):
    raise ValueError("The number of 'param_names' must match the number of 'initial_guesses'.")

# --- Global variable initialization ---
column_data = None
x_data = None
y_data = None
y_errors = None

# ==============================================================================
# 2. DYNAMIC FUNCTION CREATION (Boilerplate)
# ==============================================================================
def create_fit_function(expression_string, parameter_names):
    def fit_function(x, *params):
        param_dict = {name: value for name, value in zip(parameter_names, params)}
        local_vars = {'x': x, **param_dict}
        global_vars = {'np': np}
        return eval(expression_string, global_vars, local_vars)
    return fit_function

fit_function = create_fit_function(fit_equation_string, param_names)

# ==============================================================================
# 3. AUTOMATIC DATA LOADING 
# ==============================================================================
try:
    detected_num_columns = 0
    all_numbers = []
    
    with open(data_file, 'r') as f:
        for line in f:
            clean_line = line.strip()
            # Skip empty lines or comment lines
            if not clean_line or clean_line.startswith(('#', ';', '%')):
                print(f"Skipping line: {clean_line}")
                continue

            # Use the first valid data line to detect the number of columns
            if detected_num_columns == 0:
                detected_num_columns = len(re.split(r'[ ,\t]+', clean_line))

            # Split the line into numbers and convert to float
            numbers_on_line = [float(num) for num in re.split(r'[ ,\t]+', clean_line)]
            all_numbers.extend(numbers_on_line)
    
    if not all_numbers:
        raise ValueError("Could not find any numerical data in the file.")

    # Reshape the flat list of all numbers into the final array
    column_data = np.array(all_numbers).reshape(-1, detected_num_columns)
    
    print(f"Loaded data with {column_data.shape[0]} rows and {column_data.shape[1]} columns.")

except (IOError, ValueError, IndexError) as e:
    print(f"Error loading data: {e}")
    exit()


# =============================================================================
# 4. DATA PROCESSING (Boilerplate)
# =============================================================================
try:
    # First, we define the actual `column` function now that `column_data` exists.
    def column(i):
        """Accesses data columns by number (1-based index)."""
        if i < 1 or i > column_data.shape[1]:
            raise IndexError(f"Invalid column index: {i}. File has {column_data.shape[1]} columns.")
        return column_data[:, i-1]

    # Now, we execute the user-defined rules from the top of the script.
    print("\nProcessing data based on user-defined rules...")
    x_data, y_data, y_errors = process_data_rules(column)
    print("Data processing complete.")

except (IndexError, NameError, TypeError) as e:
    print(f"\nError during data processing: {e}")
    print("Please check your assignments inside the 'process_data_rules' function.")
    exit()


# ==============================================================================
# 5. PERFORM CURVE FIT AND DISPLAY RESULTS
# ==============================================================================
try:
    if y_errors is not None:
        print("\nPerforming weighted fit using provided errors...")
        popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses, sigma=y_errors, absolute_sigma=True)
    else:
        print("\nNo errors provided. Performing unweighted fit...")
        popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses)

    print("\nFitting complete!")
    print("Optimal parameters:")
    perr = np.sqrt(np.diag(pcov))
    result_strings = []
    for i in range(len(popt)):
        name = param_names[i]
        value = popt[i]
        error = perr[i]
        # Format the numbers using the helper function
        formatted_value = format_value(value, reporting_precision)
        formatted_error = format_value(error, reporting_precision)
        print(f"  {name} = {formatted_value} +/- {formatted_error}")
        # Create a formatted string for the plot legend
        result_strings.append(fr'{name} = {formatted_value} $\pm$ {formatted_error}')

    # --- Visualize the fit ---
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
    y_fit_line = fit_function(x_fit_line, *popt)

    plt.figure(figsize=(10, 6))
    
    if y_errors is not None:
        plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', color='blue', 
                     markersize=4, alpha=0.7, capsize=3, label='Data with Errors')
    else:
        plt.scatter(x_data, y_data, label='Data', color='blue', s=20, alpha=0.7)
    
    # Create the final legend label by joining each parameter string with a newline
    fit_label = "Fit Parameters:\n" + "\n".join(result_strings)
    
    plt.plot(x_fit_line, y_fit_line, 'r-', label=fit_label)
    
    plt.title(plot_title, fontsize=plot_title_fontsize)
    plt.xlabel(xaxis_label, fontsize=axis_label_fontsize)
    plt.ylabel(yaxis_label, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)
    plt.show()

except RuntimeError as e:
    print(f"\nCurve fitting failed: {e}")
    
    plt.figure(figsize=(10, 6))
    if y_errors is not None:
        plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', color='blue', markersize=4, capsize=3, label='Data with Errors')
    else:
        plt.scatter(x_data, y_data, label='Data', color='blue')
    
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_initial_line = fit_function(x_fit_line, *initial_guesses)
    initial_guess_label = f"Initial Guess: {', '.join([f'{n}={v:.2f}' for n, v in zip(param_names, initial_guesses)])}"
    plt.plot(x_fit_line, y_initial_line, 'g--', label=initial_guess_label)

    plt.title('FAILED FIT - Initial Guess Visualization', fontsize=plot_title_fontsize)
    plt.xlabel(xaxis_label, fontsize=axis_label_fontsize)
    plt.ylabel(yaxis_label, fontsize=axis_label_fontsize)
    plt.xticks(fontsize=tick_label_fontsize)
    plt.yticks(fontsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")