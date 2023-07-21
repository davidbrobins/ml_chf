# Test quadratic fit over XGBoost models in Z/Z_sun (with fixed feature set)
# Syntax to run python z_interp_test.py model_dir/ rate_set model_type

# Imports
import sys # Command line argument handling
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import matplotlib.pyplot as plt # Matplotlib for plotting
import CHF_gh12 # f2py wrapper module for Gnedin and Hollon 2012 CHF approximation
import configparser # Configuration file parsing module
from column_definition import * # MOdule with column name to feature name conversions
import xgboost as xgb
from pickle import load # Function to read in pickle files (like scalers)
from scipy.optimize import minimize, minimize_scalar # Numerical minimization tool from scipy
#initialize CF with values from cf_table.I2.dat
CHF_gh12.frtinitcf(0,'cf_table.I2.dat')

# Unpack command line arguments (this file, directory where models live, rate set, CF or HF)
(pyfilename, model_dir, rate_set, model_type) = sys.argv

# Define the shape of the dataframe to hold the evaluation data
eval_data = pd.DataFrame(columns = ['log10(n_H) [cm^{-3}]', 'log10(Z/Z_sun)', 'log10(u_0)', 'log10(Q_LW) [cm^3 s^{-1}]',
                                    'log10(Q_HI/Q_LW)', 'log10(Q_HeI/Q_LW)', 'log10(Q_HeII/Q_LW)', 'log10(Q_CVI/Q_LW)',
                                    'log10(Q_Al13/Q_LW)', 'log10(Q_Fe26/Q_LW)', 'log10(Q_CI/Q_LW)', 'log10(Q_C04/Q_LW)',
                                    'log10(Q_C05/Q_LW)', 'log10(Q_O06/Q_LW)', 'log10(Q_O08/Q_LW)', 'log10(Q_F09/Q_LW)',
                                    'log10(Q_Ne10/Q_LW)', 'log10(Q_Na11/Q_LW)', 'log10(Q_Mg12/Q_LW)', 'log10(Q_Si14/Q_LW)',
                                    'log10(Q_S16/Q_LW)', 'log10(Q_Ar18/Q_LW)', 'log10(Q_Ca20/Q_LW)', 'log10(Q_MgI/Q_LW)',
                                    'log10(Q_MgII/Q_LW)', 'log10(Q_MgIII/Q_LW)', 'log10(Q_MgIV/Q_LW)', 'log10(Q_MgV/Q_LW)',
                                    'log10(Q_MgVI/Q_LW)', 'log10(Q_MgVII/Q_LW)', 'log10(Q_MgVIII/Q_LW)', 'log10(Q_MgIX/Q_LW)',
                                    'log10(Q_MgX/Q_LW)', 'log10(Q_MgXI/Q_LW)', 'log10(Q_MgXII/Q_LW)', 'log10(Q_FeI/Q_LW)',
                                    'log10(Q_FeII/Q_LW)', 'log10(Q_FeIII/Q_LW)', 'log10(Q_FeIV/Q_LW)', 'log10(Q_FeV/Q_LW)',
                                    'log10(Q_FeVI/Q_LW)', 'log10(Q_FeVII/Q_LW)', 'log10(Q_FeVIII/Q_LW)', 'log10(Q_FeIX/Q_LW)',
                                    'log10(Q_FeX/Q_LW)', 'log10(Q_FeXI/Q_LW)', 'log10(Q_FeXII/Q_LW)', 'log10(Q_FeXIII/Q_LW)',
                                    'log10(Q_FeXIV/Q_LW)', 'log10(Q_FeXV/Q_LW)', 'log10(Q_FeXVI/Q_LW)', 'log10(Q_FeXVII/Q_LW)',
                                    'log10(Q_FeXVIII/Q_LW)', 'log10(Q_FeXIX/Q_LW)', 'log10(Q_FeXX/Q_LW)', 'log10(Q_FeXXI/Q_LW)',
                                    'log10(Q_FeXXII/Q_LW)', 'log10(Q_FeXXIII/Q_LW)', 'log10(Q_FeXXIV/Q_LW)', 'log10(Q_FeXXV/Q_LW)',
                                    'log10(Q_FeXXVI/Q_LW)', 'log10(T) [K]', 'CF [erg cm^3 s^{-1}]','HF [erg cm^3 s^{-1}]',
                                    'file_num'])
# Just a list of the scaled photoionization rate column names
p_rate_cols = ['log10(Q_HI/Q_LW)', 'log10(Q_HeI/Q_LW)', 'log10(Q_HeII/Q_LW)', 'log10(Q_CVI/Q_LW)',
               'log10(Q_Al13/Q_LW)', 'log10(Q_Fe26/Q_LW)', 'log10(Q_CI/Q_LW)', 'log10(Q_C04/Q_LW)', 'log10(Q_C05/Q_LW)',
               'log10(Q_O06/Q_LW)', 'log10(Q_O08/Q_LW)', 'log10(Q_F09/Q_LW)', 'log10(Q_Ne10/Q_LW)', 'log10(Q_Na11/Q_LW)',
               'log10(Q_Mg12/Q_LW)', 'log10(Q_Si14/Q_LW)', 'log10(Q_S16/Q_LW)', 'log10(Q_Ar18/Q_LW)', 'log10(Q_Ca20/Q_LW)']
p_rate_MgFe_cols = ['log10(Q_MgI/Q_LW)', 'log10(Q_MgII/Q_LW)', 'log10(Q_MgIII/Q_LW)', 'log10(Q_MgIV/Q_LW)',
                    'log10(Q_MgV/Q_LW)', 'log10(Q_MgVI/Q_LW)', 'log10(Q_MgVII/Q_LW)', 'log10(Q_MgVIII/Q_LW)', 'log10(Q_MgIX/Q_LW)',
                    'log10(Q_MgX/Q_LW)', 'log10(Q_MgXI/Q_LW)', 'log10(Q_MgXII/Q_LW)', 'log10(Q_FeI/Q_LW)', 'log10(Q_FeII/Q_LW)', 
                    'log10(Q_FeIII/Q_LW)', 'log10(Q_FeIV/Q_LW)', 'log10(Q_FeV/Q_LW)', 'log10(Q_FeVI/Q_LW)', 'log10(Q_FeVII/Q_LW)',
                    'log10(Q_FeVIII/Q_LW)', 'log10(Q_FeIX/Q_LW)', 'log10(Q_FeX/Q_LW)', 'log10(Q_FeXI/Q_LW)', 'log10(Q_FeXII/Q_LW)',
                    'log10(Q_FeXIII/Q_LW)', 'log10(Q_FeXIV/Q_LW)', 'log10(Q_FeXV/Q_LW)', 'log10(Q_FeXVI/Q_LW)', 'log10(Q_FeXVII/Q_LW)',
                    'log10(Q_FeXVIII/Q_LW)', 'log10(Q_FeXIX/Q_LW)', 'log10(Q_FeXX/Q_LW)', 'log10(Q_FeXXI/Q_LW)', 'log10(Q_FeXXII/Q_LW)', 
                    'log10(Q_FeXXIII/Q_LW)', 'log10(Q_FeXXIV/Q_LW)', 'log10(Q_FeXXV/Q_LW)', 'log10(Q_FeXXVI/Q_LW)']

# Loop through and read in files
for dir_num in range(1): # 10 for all files (plot in GH12 paper may only be for MC0 directory)
    print(dir_num)
    for file_num in range(10000): # 10000 for all files
        if file_num % 200 == 0:
            print(file_num)
        # Get filename
        file = 'data/evaluation_data/MC'+str(dir_num)+'/model.'+str(dir_num * 10000 + file_num).zfill(5)+'.res'
        # Read in log(T), CF, HF after line 3
        CHF_from_file = pd.read_csv(file, sep= '\s+', skiprows = 3, 
                                    names = ['log10(T) [K]', 'CF [erg cm^3 s^{-1}]', 'HF [erg cm^3 s^{-1}]'])
        # Get parameters, photoionization rates on first 2 lines (after 2 characters of # and space)
        with open(file) as file:
            params = file.readline().split()[3:]
            p_rates = file.readline().split()[3:]
        # First param is log(hydrogen number density), third is amplitude, last is log(metallicity/solar metallicity)
        CHF_from_file['log10(n_H) [cm^{-3}]'] = params[0]
        CHF_from_file['log10(u_0)'] = params[2]
        CHF_from_file['log10(Z/Z_sun)'] = params[-1]
        # Read additional photoionization rates
        MgFe_rates_file = 'data/evaluation_data/MC'+str(dir_num)+'/model-rates2.'+str(dir_num * 10000 + file_num).zfill(5)+'.res'
        with open(MgFe_rates_file) as file:
            redundant_params = file.readline().split()[3:]
            p_rates_MgFe = file.readline().split()[3:]
        # Check to make sure parameter sets from both files are the same
        if params != redundant_params:
            print('Uh oh! Parameters do not match!')
        # Check to make sure Lyman-Werner bands photoionization rates match
        if p_rates[0] != p_rates_MgFe[0]:
            print('Uh oh! Lyman-Werner rates do not match!')
        # Rates from files are all Q_i, need to convert all (except LW) to log(Q_i/Q_LW)
        for index in range(len(p_rate_cols)):
            CHF_from_file[p_rate_cols[index]] = np.log10(float(p_rates[1 + index])) - np.log10(float(p_rates[0]))
        for index in range(len(p_rate_MgFe_cols)):
            CHF_from_file[p_rate_MgFe_cols[index]] = np.log10(float(p_rates_MgFe[1 + index])) - np.log10(float(p_rates_MgFe[0]))
        # Convert Q_LW to log(Q_LW)
        CHF_from_file['log10(Q_LW) [cm^3 s^{-1}]'] = np.log10(float(p_rates[0]))
        # Get file number                                                                                                                                                 
        CHF_from_file['file_num'] = dir_num * 10000 + file_num
        # Concatenate the dataframe for this file onto the original
        eval_data = pd.concat([eval_data, CHF_from_file], ignore_index = True)
# Convert everything to a float
eval_data = eval_data.astype(float)
# Print length
print('All rows: ', len(eval_data.index))
# Cut rows with log(Z/Z_sun) > 0.5 (corresponds to Z > 3, outside training sample)
eval_data = eval_data.loc[eval_data['log10(Z/Z_sun)'] <= np.log10(3)]
# Print new length
print('After metallicity cut: ', len(eval_data.index))

# Loop through values of scaled Z
Z_dir_labels = ['0', '0.1', '0.3', '1', '3']
for scaled_Z in Z_dir_labels:
    print('Z/Z_sun=' + scaled_Z)
    # Assemble the appropriate directory name
    dir_name = model_dir + '/' + rate_set + '/all_data/' + model_type + '_Z_' + scaled_Z + '/'
    # Read in the scalers
    scalers = load(open(dir_name + 'scalers.pkl', 'rb'))
    # Get list of feature names from config file
    config = configparser.ConfigParser()
    config.read(dir_name + 'config.ini')
    features = [key + '_feat' for key in config['features']]
    # Loop through features
    for feature_name in features:
        # Get the appropriate scaler
        feat_scaler = scalers[feature_name]
        # Scale the feature and put it in the dataframe
        eval_data[feature_name] = feat_scaler.transform(eval_data[get_col_names(feature_name)].values.reshape(-1,1))
    print('Scaled features...')
    # Read in the XGBoost model
    model = xgb.XGBRegressor()
    model.load_model(dir_name + 'trained_model.txt')
    print('Read in model...')
    eval_data['xgb_' + model_type + '_Z_' + scaled_Z] = model.predict(eval_data[features])
    print('Trained model...')

# Interpolation in Z/Z_sun
# Create an array of the numerical values of Z/Z_sun
Z_vals = np.array([0, 0.1, 0.3, 1, 3])
# Create a column in the dataframe with Z/Z_sun
eval_data['Z/Z_sun'] = 10**eval_data['log10(Z/Z_sun)']
# Define the fitting function
def fitting_func(Z, params):
    # Get named params from array of params
    A = params[0]
    B = params[1]
    C = params[2]
    # Quadratic fit (change for other fit)
    return A + B * Z + C * (Z ** 2)
# Define a function to calculate chi-squared
def chi_squared(fit_vals, true_vals):
    # Sum squared difference over arrays (must be same length)
    return np.sum((true_vals - fit_vals) ** 2)
# Define the constraint function
def constraint(params, true_vals):
    # Get min,max allowed values (min, max true values)
    min_allowed = min(true_vals)
    max_allowed = max(true_vals)
    # For quadratic fit
    A = params[0]
    B = params[1]
    C = params[2]
    # Find vertex
    if C != 0:
        vertex = -B / (2 * C)
    else:
        vertex = np.nan
    # Find value of fitting_func on endpoints
    left = fitting_func(0, params)
    right = fitting_func(3, params)
    # If vertex is in range, it could be min/max
    if (vertex > 0) and (vertex < 3):
        min_val = min([left, right, fitting_func(vertex, params)])
        max_val = max([left, right, fitting_func(vertex, params)])
    # Otherwise, min/max values are at endpoints
    else:
        min_val = min([left, right])
        max_val = max([left, right])
    # Constraint function is satisfied (0) only if it stays within min/max of true values
    if (min_val >= min_allowed) and (max_val <= max_allowed):
        return 0
    else:
        return 1e10
# Define function to minimize
def func_to_min(params, Z_vals, true_vals):
    fit_vals = fitting_func(Z_vals, params)
    return chi_squared(fit_vals, true_vals) + constraint(params, true_vals)
# Create a wrapper function to apply the interpolation
def Z_quad_fit(input_array, Z_array = Z_vals):
    Z = input_array[0]
    # Do fit in CF/HF (not log)
    model_along_Z_array = 10**input_array[1:]
    # Minimize func_to_min with appropriate arguments
    x0 = [model_along_Z_array[0],
          1/6*(-model_along_Z_array[4] + 9*model_along_Z_array[3] - 8*model_along_Z_array[0]),
          1/6*(model_along_Z_array[4] - 3*model_along_Z_array[3] + 2*model_along_Z_array[0])]
    initial_simplex = [x0, [model_along_Z_array[0], 0, 0],
                       [model_along_Z_array[0], 1/3*(model_along_Z_array[4] - model_along_Z_array[0]), 0],
                       [model_along_Z_array[0], 10*(model_along_Z_array[1] - model_along_Z_array[0]), 0]]
    fit_params_a = minimize(func_to_min, x0, method = 'nelder-mead', args = (Z_array, model_along_Z_array), options = {'initial_simplex' : initial_simplex}).x
    print('First optimization: ', fit_params_a, func_to_min(fit_params_a, Z_array, model_along_Z_array))
    are_params_close = False
    counter = 1
    while not are_params_close:
        new_simplex = [fit_params_a, [(1 + np.random.normal()) * fit_params_a[0], fit_params_a[1], fit_params_a[2]],
                       [fit_params_a[0], (1 + np.random.normal()) * fit_params_a[1], fit_params_a[2]],
                       [fit_params_a[0], fit_params_a[1], (1 + np.random.normal()) * fit_params_a[2]]]
        fit_params_b = minimize(func_to_min, fit_params_a, method = 'nelder-mead', args = (Z_array, model_along_Z_array), options = {'initial_simplex' : new_simplex}).x
        are_params_close = np.allclose(fit_params_a, fit_params_b, atol = 0)
        fit_params_a = fit_params_b
        counter += 1
        print('Iteration ' + str(counter) + ': ', fit_params_a, func_to_min(fit_params_a, Z_array, model_along_Z_array))
    return fit_params_a[0], fit_params_a[1], fit_params_a[2], fit_params_a[0] + fit_params_a[1] * Z + fit_params_a[2] * (Z ** 2)
    '''
    # Set up matrices for minimizing chi squared
    M = np.array([[np.sum(Z_array**4), np.sum(Z_array**3), np.sum(Z_array**2)],
                  [np.sum(Z_array**3), np.sum(Z_array**2), np.sum(Z_array)],
                  [np.sum(Z_array**2), np.sum(Z_array), len(Z_array)]])
    N = np.array([np.sum((model_along_Z_array) * Z_array * Z_array),
                  np.sum((model_along_Z_array) * Z_array),
                  np.sum(model_along_Z_array)])
    # Get coefficients from solving Mx=N, where x = [A B]
    coeffs = np.linalg.solve(M, N)
    return 10**(coeffs[0]* Z**2 + coeffs[1] * Z + coeffs[2])
    '''
# Do quadratic fit in Z/Z_sun
eval_data[['A', 'B', 'C', 'xgb_quad_fit']] = eval_data[['Z/Z_sun'] + ['xgb_' + model_type + '_Z_' + scaled_Z for scaled_Z in Z_dir_labels]].apply(Z_quad_fit, axis = 1, result_type = 'expand')
print('Did quadratic fit in Z...')

# Define a helper function to calculate error in log(CHF), and return 1000 if CHF prediction is negative                                                                         
def get_err(input_array):
    # Input array is of the form [thruth, prediction]                                                                                                                            
    truth = input_array[0]
    pred = input_array[1]
    # If prediction is negative ('catastrophic error'), return large error of 1000                                                                                              
    if pred <= 0:
        return 1000
    # Otherwise, calculate |log(truth/prediction)|                                                                                                                               
    else:
        return np.abs(np.log10(truth/pred))
# Calculate errors for these interpolations
eval_data['xgb_quad_err'] = eval_data[[model_type + ' [erg cm^3 s^{-1}]', 'xgb_quad_fit']].apply(get_err, axis = 1)

# Get sorted arrays of errors in CF, HF, max
err_quad_Z = np.sort(eval_data['xgb_quad_err'].values)
# Length of all of this is just length of index of eval_data
length = len(eval_data.index)
# Get 1 - CDF
freq = (length - np.arange(length)) / length
print('Got CDF')

'''
# Save it to a file
xgb_err_cdf = pd.DataFrame(index = range(len(freq)), columns = ['err_interp_Z', 'err_interp_Z_2', 'err_interp_Z_0.5', 'freq'])
xgb_err_cdf['err_interp_Z'] = err_interp_Z
xgb_err_cdf['err_interp_Z_2'] = err_interp_Z_2
xgb_err_cdf['err_interp_Z_0.5'] = err_interp_Z_0p5
xgb_err_cdf['freq'] = freq
xgb_err_cdf.to_pickle(model_dir + '/' + rate_set + '/' + model_type + '_interp_cdf.pkl')
'''

# Read in GH12 CDF
gh12_err_cdf = load(open('data/evaluation_data/gh12_err_cdf.pkl', 'rb'))

# Define step size for downsampling
step = 1
plt.plot(gh12_err_cdf[model_type + '_err_gh12'][::step], gh12_err_cdf['freq'][::step], linestyle = 'solid', label = 'GH12')
plt.plot(err_quad_Z[::step], freq[::step], linestyle = 'dashed', label = 'XGBoost + quadratic fit')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.01, 2]) # upper lim = 2 to match Gnedin and Hollon 2012 paper
plt.ylim([1e-6, 1])
plt.xlabel('$\Delta \log \mathcal{F}$')
plt.ylabel('$F(>\Delta \log \mathcal{F})$')
plt.title(model_type + ', ' + rate_set)
plt.legend()
#plt.savefig(model_dir + '/' + rate_set + '/' + model_type + '_err_cdf_test.pdf')
# Just for first test
plt.savefig('quad_fit_test.png')
plt.close()
print('Made error CDF plots')
print('Done!')


# Get lines with error above 1
high_errs = eval_data.loc[eval_data['xgb_quad_err'] > 1]
high_errs = high_errs[['Z/Z_sun', model_type + ' [erg cm^3 s^{-1}]', 'file_num', 'log10(T) [K]', 'A', 'B', 'C', 'xgb_quad_fit'] + ['xgb_' + model_type + '_Z_' + scaled_Z for scaled_Z in Z_dir_labels]]
print(high_errs)

for i in range(10):                                                                                                                                                              
    CF_vs_Z = high_errs.iloc[i]                                                                                                                                              
    print(CF_vs_Z)                                                                                                                                                               
    plt.scatter(Z_vals, 10**CF_vs_Z[['xgb_' + model_type + '_Z_' + scaled_Z for scaled_Z in Z_dir_labels]], color = 'orange', label = 'XGBoost')                           
    plt.scatter(CF_vs_Z['Z/Z_sun'], CF_vs_Z[model_type + ' [erg cm^3 s^{-1}]'], color = 'green', label = 'True value')
    # Create array of Z value to plot quadratic at
    plot_Z = np.linspace(0, 3, 100)
    plt.plot(plot_Z, CF_vs_Z['A'] + CF_vs_Z['B'] * plot_Z + CF_vs_Z['C'] * plot_Z ** 2, color = 'orange', label = 'Quadratic fit')
    plt.legend()                                                                                                                                                                 
    plt.xlabel('$Z/Z_\odot$')                                                                                                                                                    
    plt.ylabel(model_type + ' [erg cm^3 s^{-1}]')                                                                                                                                
    plt.savefig('CF_vs_Z_test_' + str(i) + '.png')                                                                                                                               
    plt.close()   

