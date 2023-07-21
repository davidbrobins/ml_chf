# Evaluate and save GH12 interpolation table error distributions on off-grid evaluation data

# Imports
import sys # Command line argument handling
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import matplotlib.pyplot as plt # Matplotlib for plotting
import CHF_gh12 # f2py wrapper module for Gnedin and Hollon 2012 CHF approximation
#initialize CF with values from cf_table.I2.dat
CHF_gh12.frtinitcf(0,'cf_table.I2.dat')

# Define the shape of the dataframe to hold the evaluation data
eval_data = pd.DataFrame(columns = ['log10(n_H [cm^{-3}])', 'log10(Z/Z_sun)', 'log10(u_0)', 'log10(Q_LW) [cm^3 s^{-1}]',
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
                                    'log10(Q_FeXXVI/Q_LW)', 'log10(T [K])', 'CF [erg cm^3 s^{-1}]','HF [erg cm^3 s^{-1}]',
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
                                    names = ['log10(T [K])', 'CF [erg cm^3 s^{-1}]', 'HF [erg cm^3 s^{-1}]'])
        # Get parameters, photoionization rates on first 2 lines (after 2 characters of # and space)
        with open(file) as file:
            params = file.readline().split()[3:]
            p_rates = file.readline().split()[3:]
        # First param is log(hydrogen number density), third is amplitude, last is log(metallicity/solar metallicity)
        CHF_from_file['log10(n_H [cm^{-3}])'] = params[0]
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
# Cut rows with log(Z/Z_sun) > log(3), removes rows with higher metallicity than in training sample
eval_data = eval_data.loc[eval_data['log10(Z/Z_sun)'] <= np.log10(3)]
# Print new length
print('After metallicity cut: ', len(eval_data.index))

# Define the w function for converting between n_H and n_b
# Note: n_H = w*n_b
def w(Z):
    '''
    Function to give metallicity-dependent conversion factor between n_H and n_b, where n_H=w*n_b.
    Input:
    Z (float): Metallicity Z, in units of Z_sun
    Output:
    w (float): w(Z) = (1.0-0.02*Z)/1.4
    '''
    
    return (1.0 - 0.02 * Z) / 1.4

# Define helper functions to return cooling and heating functions
def get_cf(input_array):
    (cfun, hfun, ierr) = CHF_gh12.frtgetcf(*input_array)
    return cfun
def get_hf(input_array):
    (cfun, hfun, ierr) = CHF_gh12.frtgetcf(*input_array)
    return hfun

# Calculate Z/Zsun, n_H, T from logs
eval_data['Z/Z_sun'] = 10 ** eval_data['log10(Z/Z_sun)']
eval_data['n_H [cm^{-3}]'] = 10 ** eval_data['log10(n_H [cm^{-3}])']
eval_data['T [K]'] = 10 ** eval_data['log10(T [K])']
# Convert hydrogen number density to baryon number density using w(Z) factor
eval_data['n_b [cm^{-3}]'] = 1.0/w(eval_data['Z/Z_sun']) * eval_data['n_H [cm^{-3}]']
# Calculate the 4 P_i we need for GH12 approximation, where Q_i = P_i / n_H
eval_data['P_LW [s^{-1}]'] = (10 ** eval_data['log10(Q_LW) [cm^3 s^{-1}]']) * eval_data['n_H [cm^{-3}]']
eval_data['P_HI [s^{-1}]'] = (10 ** eval_data['log10(Q_HI/Q_LW)']) * eval_data['P_LW [s^{-1}]']
eval_data['P_HeI [s^{-1}]'] = (10 ** eval_data['log10(Q_HeI/Q_LW)']) * eval_data['P_LW [s^{-1}]']
eval_data['P_CVI [s^{-1}]'] = (10 ** eval_data['log10(Q_CVI/Q_LW)']) * eval_data['P_LW [s^{-1}]']
# Calculate CF, HF columns (use BARYON number density for approximation) 
eval_data['CF_gh12 [erg cm^3 s^{-1}]'] = eval_data[['T [K]', 'n_b [cm^{-3}]', 'Z/Z_sun', 'P_LW [s^{-1}]', 
                                                    'P_HI [s^{-1}]', 'P_HeI [s^{-1}]', 'P_CVI [s^{-1}]']].apply(get_cf, axis = 1)
eval_data['HF_gh12 [erg cm^3 s^{-1}]'] = eval_data[['T [K]', 'n_b [cm^{-3}]', 'Z/Z_sun', 'P_LW [s^{-1}]', 
                                                    'P_HI [s^{-1}]', 'P_HeI [s^{-1}]', 'P_CVI [s^{-1}]']].apply(get_hf, axis = 1)
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
        
# Get error in log(CHF) for GH12 approx
eval_data['cf_err_gh12'] = eval_data[['CF [erg cm^3 s^{-1}]', 'CF_gh12 [erg cm^3 s^{-1}]']].apply(get_err, axis = 1)
eval_data['hf_err_gh12'] = eval_data[['HF [erg cm^3 s^{-1}]', 'HF_gh12 [erg cm^3 s^{-1}]']].apply(get_err, axis = 1)
print('Evaluated GH 12 approx')

# Get columns of error in the max of CF, HF
eval_data['max_err_gh12'] = eval_data['cf_err_gh12'] * (eval_data['CF [erg cm^3 s^{-1}]'] > eval_data['HF [erg cm^3 s^{-1}]']) + eval_data['hf_err_gh12'] * (1 - (eval_data['CF [erg cm^3 s^{-1}]'] > eval_data['HF [erg cm^3 s^{-1}]']))

# Get array of unique log(T) values
temps = eval_data['log10(T [K])'].unique()
# Blank arrays for 50th, 90th, 99th error percentiles
CF_med_gh12 = []
HF_med_gh12 = []
max_med_gh12 = []
CF_90_gh12 = []
HF_90_gh12 = []
max_90_gh12 = []
CF_99_gh12 = []
HF_99_gh12 = []
max_99_gh12 = []
# Loop through the unique temperatures
for temp in temps:
    # Get all rows with this temp
    temp_data = eval_data.loc[eval_data['log10(T [K])'] == temp]
    # Get error values
    CF_errs_gh12 = temp_data['cf_err_gh12'].values
    HF_errs_gh12 = temp_data['hf_err_gh12'].values
    max_errs_gh12 = temp_data['max_err_gh12'].values
    # Append statistics to the appropriate arrays
    CF_med_gh12.append(np.median(CF_errs_gh12))
    CF_90_gh12.append(np.percentile(CF_errs_gh12, 90))
    CF_99_gh12.append(np.percentile(CF_errs_gh12, 99))
    HF_med_gh12.append(np.median(HF_errs_gh12))
    HF_90_gh12.append(np.percentile(HF_errs_gh12, 90))
    HF_99_gh12.append(np.percentile(HF_errs_gh12, 99))
    max_med_gh12.append(np.median(max_errs_gh12))
    max_90_gh12.append(np.percentile(max_errs_gh12, 90))
    max_99_gh12.append(np.percentile(max_errs_gh12, 99))
    print('Did log10(T [K])=', temp)

# Put all of this into a dataframe and save it
gh12_err_vs_T = pd.DataFrame(index = range(len(temps)), columns = ['T [K]', 'CF_med_gh12', 'CF_90_gh12', 'CF_99_gh12', 'HF_med_gh12', 'HF_90_gh12', 'HF_99_gh12', 'max_med_gh12', 'max_90_gh12', 'max_99_gh12'])
gh12_err_vs_T['T [K]'] =  temps
gh12_err_vs_T['CF_med_gh12'] = CF_med_gh12
gh12_err_vs_T['CF_90_gh12'] = CF_90_gh12
gh12_err_vs_T['CF_99_gh12'] = CF_99_gh12
gh12_err_vs_T['HF_med_gh12'] = HF_med_gh12
gh12_err_vs_T['HF_90_gh12'] = HF_90_gh12
gh12_err_vs_T['HF_99_gh12'] = HF_99_gh12
gh12_err_vs_T['max_med_gh12'] = max_med_gh12
gh12_err_vs_T['max_90_gh12'] = max_90_gh12
gh12_err_vs_T['max_99_gh12'] = max_99_gh12
gh12_err_vs_T.to_pickle('data/evaluation_data/gh12_err_vs_T.pkl')

# Plot for gh12
plt.semilogy(temps, CF_med_gh12, linestyle = 'solid', color = 'blue')
plt.semilogy(temps, HF_med_gh12, linestyle = 'solid', color = 'red')
plt.semilogy(temps, max_med_gh12, linestyle = 'solid', color = 'black')
plt.semilogy(temps, CF_90_gh12, linestyle = 'dashed', color = 'blue')
plt.semilogy(temps, HF_90_gh12, linestyle = 'dashed', color = 'red')
plt.semilogy(temps, max_90_gh12, linestyle = 'dashed', color = 'black')
plt.semilogy(temps, CF_99_gh12, linestyle = 'dotted', color = 'blue')
plt.semilogy(temps, HF_99_gh12, linestyle = 'dotted', color = 'red')
plt.semilogy(temps, max_99_gh12, linestyle = 'dotted', color = 'black')
plt.xlabel('$\log{(T \, [\mathrm{K}])}$')
plt.ylabel('$\Delta \log \Gamma$')
plt.savefig('gh12_err_vs_T_test.pdf')
plt.close()
print('Made error vs. T plots')

# Get sorted arrays of errors in CF, HF, max
CF_err_gh12 = np.sort(eval_data['cf_err_gh12'].values)
HF_err_gh12 = np.sort(eval_data['hf_err_gh12'].values)
max_err_gh12 = np.sort(eval_data['max_err_gh12'].values)
# Length of all of these is just length of index of eval_data
length = len(eval_data.index)
# Get 1 - CDF
freq = (length - np.arange(length)) / length
print('Got CDF')
# Put this data in a dataframe and save it
gh12_err_cdf = pd.DataFrame(index = range(len(freq)), columns = ['CF_err_gh12', 'HF_err_gh12', 'max_err_gh12', 'freq'])
gh12_err_cdf['CF_err_gh12'] = CF_err_gh12
gh12_err_cdf['HF_err_gh12'] = HF_err_gh12
gh12_err_cdf['max_err_gh12'] = max_err_gh12
gh12_err_cdf['freq'] = freq
gh12_err_cdf.to_pickle('data/evaluation_data/gh12_err_cdf.pkl')

# Define step size for downsampling
step = 1
plt.plot(CF_err_gh12[::step], freq[::step], color = 'blue', linestyle = 'solid', label = '$\Lambda$')
plt.plot(HF_err_gh12[::step], freq[::step], color = 'red', linestyle = 'solid', label = '$\Gamma$')
plt.plot(max_err_gh12[::step], freq[::step], color = 'black', linestyle = 'solid', label = 'max, GH12')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.01, 2]) # upper lim = 2 to match Gnedin and Hollon 2012 paper
plt.ylim([1e-6, 1])
plt.xlabel('$\Delta \log \mathcal{F}$')
plt.ylabel('$F(>\Delta \log \mathcal{F})$')
plt.legend()
plt.savefig('err_cdf_test_gh12.pdf')
print('Made error CDF plot')
print('Done!')

# Get rows where CHF err is greater than 1 for GH12 approx
high_errs = eval_data.loc[(eval_data['cf_err_gh12'] >= 1) | (eval_data['hf_err_gh12'] >= 1)]
# Print the error, temperature, and file number for those rows
print(high_errs[['log10(T [K])', 'cf_err_gh12', 'hf_err_gh12', 'file_num']])
