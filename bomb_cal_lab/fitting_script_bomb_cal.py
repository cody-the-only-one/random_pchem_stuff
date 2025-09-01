import numpy as np
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
import re
# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
# This is the only section you should need to edit for a new analysis.

# --- Data and Plotting ---
data_file = '/content/lab_data/test1'
xaxis_label = 'X-axis Label'
yaxis_label = 'Y-axis Label'
plot_title = 'Function Fit with Residuals'

# --- Fitting Function Definition ---

# Independent variable is time
independent_variable = 't' 
# The composite function modeling the entire calorimeter curve
#fit_equation_string = "(T0 + m_pre*(t - t0)) + (dT + (m_post - m_pre)*(t - t0)) / (1 + np.exp(-k*(t - t0)))"
fit_equation_string = "(T0 + m_pre*(t - t0)) + (dT + (-0.0001 - m_pre)*(t - t0)) / (1 + np.exp(-k*(t - t0)))"

# Names of the parameters to be fitted
#param_names = ['t0', 'T0', 'dT', 'm_pre', 'm_post', 'k']
param_names = ['t0', 'T0', 'dT', 'm_pre', 'k']
# You will need to provide good initial guesses based on your data!
# Example:
# t0: The time you pressed the ignition button (e.g., 300 seconds)
# T0: The temperature right before ignition (e.g., 24.5 C)
# dT: The approximate raw temperature rise (e.g., 2.0 C)
# m_pre: The slope of the first few points (e.g., 1e-4 C/s)
# m_post: The slope of the last few points (e.g., -5e-4 C/s)
# k: The steepness of the rise, often between 0.1 and 1 (e.g., 0.5)
#initial_guesses = [300, 22.5, 3.0, 1e-4, -5e-4, 0.1]
initial_guesses = [300, 22.5, 3.0, 1e-4, 0.1]

# --- Data Processing Rules ---
def process_data_rules(column):
    # --- Define x and y data ---
    x_data = column(1) 
    y_data = column(2)
    
    # --- Define x_errors (optional, for ODR fit) ---
    # To use ODR, you MUST provide x_errors (not None). Otherwise, a standard fit is used.
    x_errors = None             # Leave as `None` for a standard fit without x-errors.
    # x_errors = 0.1            # Use a constant error for all x points

    # --- Define y_errors to enable a weighted fit (optional) ---
    y_errors = 0.02             # Use a constant error for all points
    
    return x_data, y_data, x_errors, y_errors

# ---- Plotting Style variables ----
plot_title_fontsize = 18
axis_label_fontsize = 14
legend_fontsize = 12
tick_label_fontsize = 12
reporting_precision = 2 
legend_location = 'best'


# ==============================================================================
#           END OF USER CONFIGURATION - NO NEED TO EDIT BELOW THIS LINE
# ==============================================================================
def format_value(value, precision):
    if value == 0: return f"{0.0:.{precision}f}"
    abs_val = abs(value)
    if abs_val > 1e5 or abs_val < 0.1: return f"{value:.{precision}e}"
    else: return f"{value:.{precision}f}"

if len(param_names) != len(initial_guesses):
    raise ValueError("The number of 'param_names' must match the number of 'initial_guesses'.")

column_data, x_data, y_data, x_errors, y_errors = None, None, None, None, None

# ==============================================================================
# 2. DYNAMIC FUNCTION CREATION (Boilerplate)
# ==============================================================================
def create_curve_fit_function(expression_string, parameter_names, indep_var_name='x'):
    def fit_function(indep_var_data, *params):
        param_dict = {name: value for name, value in zip(parameter_names, params)}
        local_vars = {indep_var_name: indep_var_data, **param_dict}
        global_vars = {'np': np}
        return eval(expression_string, global_vars, local_vars)
    return fit_function

def create_odr_fit_function(expression_string, parameter_names, indep_var_name='x'):
    def odr_function(B, x_data):
        param_dict = {name: value for name, value in zip(parameter_names, B)}
        local_vars = {indep_var_name: x_data, **param_dict}
        global_vars = {'np': np}
        return eval(expression_string, global_vars, local_vars)
    return odr_function

curve_fit_function = create_curve_fit_function(fit_equation_string, param_names, independent_variable)
odr_fit_function = create_odr_fit_function(fit_equation_string, param_names, independent_variable)

# ==============================================================================
# 3. AUTOMATIC DATA LOADING 
# ==============================================================================
try:
    detected_num_columns, all_numbers = 0, []
    with open(data_file, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith(('#', ';', '%')): continue
            if detected_num_columns == 0:
                split_line = [item for item in re.split(r'[ ,\t]+', clean_line) if item]
                detected_num_columns = len(split_line)
            numbers_on_line = [float(num) for num in re.split(r'[ ,\t]+', clean_line) if num]
            all_numbers.extend(numbers_on_line)
    if not all_numbers: raise ValueError("Could not find any numerical data in the file.")
    if len(all_numbers) % detected_num_columns != 0: raise ValueError(f"Inconsistent number of columns detected.")
    column_data = np.array(all_numbers).reshape(-1, detected_num_columns)
    print(f"Loaded data with {column_data.shape[0]} rows and {column_data.shape[1]} columns.")
except (IOError, ValueError, IndexError) as e: print(f"Error loading data: {e}"); exit()

# =============================================================================
# 4. DATA PROCESSING (Boilerplate)
# =============================================================================
try:
    def column(i):
        if i < 1 or i > column_data.shape[1]: raise IndexError(f"Invalid column index: {i}. File has {column_data.shape[1]} columns.")
        return column_data[:, i-1]
    print("\nProcessing data based on user-defined rules...")
    x_data, y_data, x_errors, y_errors = process_data_rules(column)
    print("Data processing complete.")
except (IndexError, NameError, TypeError) as e: print(f"\nError during data processing: {e}\nPlease check your assignments inside the 'process_data_rules' function."); exit()

# ==============================================================================
# 5. PERFORM CURVE FIT AND DISPLAY RESULTS
# ==============================================================================
try:
    popt, pcov = None, None

    if x_errors is not None:
        print("\nX errors provided. Performing Orthogonal Distance Regression (ODR) fit...")
        odr_model = Model(odr_fit_function)
        odr_data = RealData(x_data, y_data, sx=x_errors, sy=y_errors)
        odr_instance = ODR(odr_data, odr_model, beta0=initial_guesses)
        odr_output = odr_instance.run()
        popt = odr_output.beta
        pcov = odr_output.cov_beta
        print("ODR fit complete!")
    else:
        if y_errors is not None:
            print("\nPerforming weighted least-squares fit (errors in Y only)...")
            popt, pcov = curve_fit(curve_fit_function, x_data, y_data, p0=initial_guesses, sigma=y_errors, absolute_sigma=True)
        else:
            print("\nNo errors provided. Performing unweighted least-squares fit...")
            popt, pcov = curve_fit(curve_fit_function, x_data, y_data, p0=initial_guesses)
        print("curve_fit complete!")

    print("\nOptimal parameters:")
    perr = np.sqrt(np.diag(pcov))
    result_strings = []
    for i in range(len(popt)):
        name, value, error = param_names[i], popt[i], perr[i]
        formatted_value = format_value(value, reporting_precision)
        formatted_error = format_value(error, reporting_precision)
        print(f"  {name} = {formatted_value} +/- {formatted_error}")
        result_strings.append(fr'{name} = {formatted_value} $\pm$ {formatted_error}')

    # --- Visualize the fit with residuals ---
    if x_errors is not None:
        # Use ODR function signature: f(params, x_data)
        y_fit_at_data_points = odr_fit_function(popt, x_data)
        x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit_line = odr_fit_function(popt, x_fit_line)
    else:
        # Use curve_fit function signature: f(x_data, *params)
        y_fit_at_data_points = curve_fit_function(x_data, *popt)
        x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit_line = curve_fit_function(x_fit_line, *popt)
        
    residuals = y_data - y_fit_at_data_points
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle(plot_title, fontsize=plot_title_fontsize)

    # Plot 1: Main data and fit line
    ax1.errorbar(x_data, y_data, xerr=x_errors, yerr=y_errors, fmt='o', color='blue', markersize=4, alpha=0.7, capsize=3, label='Data with Errors')
    
    # Restored your original, more robust formatting method
    display_equation = fit_equation_string.replace('*', r' \cdot ')
    display_equation = display_equation.replace('np.exp', r'\exp').replace('np.sin', r'\sin').replace('np.cos', r'\cos')
    display_equation = display_equation.replace('np.log', r'\ln').replace('np.sqrt', r'\sqrt')
    
    fit_label = f"Fit Equation: ${display_equation}$\n\n" + "Parameters:\n" + "\n".join(result_strings)
    
    ax1.plot(x_fit_line, y_fit_line, 'r-', label=fit_label)
    ax1.set_ylabel(yaxis_label, fontsize=axis_label_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_label_fontsize)
    ax1.legend(loc=legend_location, fontsize=legend_fontsize)
    ax1.grid(True)

    # Plot 2: Residuals
    ax2.errorbar(x_data, residuals, xerr=x_errors, yerr=y_errors, fmt='o', color='green', markersize=4, alpha=0.7, capsize=3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel(xaxis_label, fontsize=axis_label_fontsize)
    ax2.set_ylabel("Residuals (y-y_fit)", fontsize=axis_label_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_label_fontsize)
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except RuntimeError as e:
    print(f"\nCurve fitting failed: {e}")
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_data, y_data, xerr=x_errors, yerr=y_errors, fmt='o', color='blue', markersize=4, capsize=3, label='Data with Errors')
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_initial_line = curve_fit_function(x_fit_line, *initial_guesses)
    initial_guess_label = f"Initial Guess: {', '.join([f'{n}={v:.2f}' for n, v in zip(param_names, initial_guesses)])}"
    plt.plot(x_fit_line, y_initial_line, 'g--', label=initial_guess_label)
    plt.title('FAILED FIT - Initial Guess Visualization', fontsize=plot_title_fontsize)
    plt.xlabel(xaxis_label, fontsize=axis_label_fontsize)
    plt.ylabel(yaxis_label, fontsize=axis_label_fontsize)
    plt.legend(loc=legend_location, fontsize=legend_fontsize)
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")