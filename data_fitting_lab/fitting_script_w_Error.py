# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
# This is the only section you should need to edit for a new analysis.

# --- Data and Plotting ---
data_file = '/content/sample_data/test1'
xaxis_label = 'X-axis Label'
yaxis_label = 'Y-axis Label'
plot_title = 'Function Fit'

# --- Fitting Function Definition ---
# Define the equation as a string. Use 'x' for the independent variable.
fit_equation_string = "a * np.exp(b * x) + c"
param_names = ['a', 'b', 'c']
initial_guesses = [10.0, -1.0, 42.0]

# --- Data Processing Rules ---
def process_data_rules(column):
    # --- Define x and y data ---
    # `column(1)` is the first column, `column(2)` is the second, etc.
    x_data = column(1) 
    y_data = column(2)
    # y_data = column(2)**2/237.238 # or do math
    
    # --- Define y_errors to enable a weighted fit (optional) ---
    #y_errors = None              # Leave as `None` for a standard, unweighted fit.
    y_errors = column(1)       # Use the 3rd column from your file for errors
    # y_errors = 2.0             # Use a constant error for all points
    # y_errors = np.sqrt(y_data) # Calculate Poisson error from y-data
    
    return x_data, y_data, y_errors

# ==============================================================================
#           END OF USER CONFIGURATION - NO NEED TO EDIT BELOW THIS LINE
# ==============================================================================

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
# 3. AUTOMATIC DATA LOADING (Boilerplate)
# ==============================================================================
try:
    detected_num_columns = 0
    with open(data_file, 'r') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith(('#', ';', '%')):
                detected_num_columns = len(re.split(r'[ ,\t]+', clean_line))
                break
    if detected_num_columns == 0:
        raise ValueError("Could not find any data in the file.")

    with open(data_file, 'r') as f:
        file_content = f.read()
    data = [float(num) for num in re.split(r'[ ,\t\n]+', file_content.strip()) if num]
    column_data = np.array(data).reshape(-1, detected_num_columns)
    
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
    # --- Change 4: Perform weighted or unweighted fit based on y_errors ---
    if y_errors is not None:
        print("\nPerforming weighted fit using provided errors...")
        # absolute_sigma=True is crucial for getting meaningful parameter errors.
        popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses, sigma=y_errors, absolute_sigma=True)
    else:
        print("\nNo errors provided. Performing unweighted fit...")
        popt, pcov = curve_fit(fit_function, x_data, y_data, p0=initial_guesses)

    # --- Print and interpret the results dynamically ---
    print("\nFitting complete!")
    print("Optimal parameters:")
    perr = np.sqrt(np.diag(pcov))

    result_strings = []
    for i in range(len(popt)):
        name, value, error = param_names[i], popt[i], perr[i]
        print(f"  {name} = {value:.4f} +/- {error:.4f}")
        result_strings.append(f"{name}={value:.2f}")

    # --- Visualize the fit ---
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 200)
    y_fit_line = fit_function(x_fit_line, *popt)

    plt.figure(figsize=(10, 6))
    
    # --- Change 4: Plot with or without error bars ---
    if y_errors is not None:
        plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', color='blue', 
                     markersize=4, alpha=0.7, capsize=3, label='Data with Errors')
    else:
        plt.scatter(x_data, y_data, label='Data', color='blue', s=20, alpha=0.7)
    
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
    # (The rest of the error handling remains the same, but now includes error bars if available)
    plt.figure(figsize=(10, 6))
    if y_errors is not None:
        plt.errorbar(x_data, y_data, yerr=y_errors, fmt='o', color='blue', markersize=4, capsize=3, label='Data with Errors')
    else:
        plt.scatter(x_data, y_data, label='Data', color='blue')
    # ... (rest of the failure plot is the same)
    x_fit_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_initial_line = fit_function(x_fit_line, *initial_guesses)
    initial_guess_label = f"Initial Guess: {', '.join([f'{n}={v:.2f}' for n, v in zip(param_names, initial_guesses)])}"
    plt.plot(x_fit_line, y_initial_line, 'g--', label=initial_guess_label) # Green dashed line for guess
    plt.title('FAILED FIT - Initial Guess Visualization')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")