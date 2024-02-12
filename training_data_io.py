# Module handling reading in the needed subset of training data as a dataframe

import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math

def series_to_list(series):
    '''
    Helper function to convert a Pandas series to a python list
    Input:
    series (pandas.Series): Series to convert to a list.
    Output:
    list (list): The converted list.
    '''

    # Do the conversion, return the result
    return list(series.values)

def get_training_data(data_path, target, output, Z_vals):
    '''
    Function to read in the needed susbset of training data, given relevant parameters from the config file.
    Input:
    data_path (str): Path to the directory containing the training data.
    target (str): The name of the target column
    output (str): Which of CF or HF is to be predicted.
    Z_vals (numpy array): Array containing the values of Z/Z_sun to read in
    Output:
    data_df (dataframe): The desired subset of the training data.
    '''

    # Binned RF column names
    rf_bin_cols = ['log10(f_q)', 'log10(tau_0)', '0_to_1_Ry', '1_to_4_Ry', '4_to_7_Ry', 
                   '7_to_10_Ry', '10_to_13_Ry', '13_to_16_Ry', '16_to_19_Ry', '19_to_22_Ry'] 
    # CHF data column names
    chf_cols = ['log10(n_H) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
                'log10(f_q)', 'log10(tau_0)', 'CF_Z_0','CF_Z_0.1', 'CF_Z_0.3', 'CF_Z_1', 'CF_Z_3',
                'HF_Z_0', 'HF_Z_0.1', 'HF_Z_0.3', 'HF_Z_1', 'HF_Z_3'] 
    # Create a blank dataframe to store the dataframes for each alpha
    alpha_dfs = {}
    # Define list of alpha values to loop through
    alpha_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Loop through alpha values
    for alpha in alpha_vals:
        # Read in photoionization rate data at this alpha value
        binned_rfs = pd.read_csv(data_path + '/rf_bin_data/pratesLin3_'+str(alpha)+'.dat', sep= '\s+', names = rf_bin_cols)
        
        # Read in CHF data at this alpha value
        chf = pd.read_csv(data_path + '/raw_data/raw_'+str(alpha)+'.res', sep='\s+',
                          usecols = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20], names = chf_cols)
        # Merge the chf and p_rates_all dataframes on the matching columns (fq, tau0)
        merged = chf.merge(binned_rfs, on = ['log10(f_q)', 'log10(tau_0)'])
        # Scale average RF in each bin by 10**J0
        for col in rf_bin_cols[2:]:
            merged[col] = merged[col] * (10 ** merged['log10(J_0/n_b/J_{MW})'])
        
        # Add a column with the alpha value
        merged['alpha'] = alpha

        # Put the merged dataframe in the dictionary
        alpha_dfs[alpha] = merged

    # Merge dataframes for all the alpha values
    data_df = pd.concat([alpha_dfs[alpha] for alpha in alpha_vals])
                                                                                  
    # If only one Z value
    if len(Z_vals) == 1:
        # Get Z value
        Z = Z_vals[0]
        # Create columns containing log10(CF), log10(HF) at Z/Z_sun value
        data_df['log10(CF) [erg cm^{3} s^{-1}]'] = np.log10(data_df['CF_Z_' + str(Z)])
        data_df['log10(HF) [erg cm^{3} s^{-1}]'] = np.log10(data_df['HF_Z_' + str(Z)])
        # If Z=0, convert to real value of 1e-4
        if Z == 0:
            Z = 1e-4
        # Make column with the fixed Z/Z_sun value
        data_df['log10(Z/Z_sun)'] = np.log10(Z)
    # Otherwise, there are multiple Z values
    else:
        # Create columns containing arrays of log10(CF), log10(HF) at Z/Z_sun values in Z_vals                                                        
        # To get these, apply series -> list helper function on CF/HF(Z) columns in each row
        data_df['log10(CF) [erg cm^{3} s^{-1}]'] = np.log10(data_df[['CF_Z_' + str(val) for val in Z_vals]]).apply(series_to_list, axis = 1)
        data_df['log10(HF) [erg cm^{3} s^{-1}]'] = np.log10(data_df[['HF_Z_' + str(val) for val in Z_vals]]).apply(series_to_list, axis = 1)
        # Convert first Z/Z_sun value to 1e-4 (actual value) from 0
        Z_vals[0] = 1e-4
        # Create a column which contains the values of log(Z/Z_sun) used
        data_df['log10(Z/Z_sun)'] = pd.Series([np.log10(Z_vals) for x in range(len(data_df.index))])
        # Now, expand these 3 aligned lists, with all elements getting their own row (and only take the columns we'll need later)
        data_df = data_df[['log10(n_H) [cm^{-3}]', 'log10(T) [K]', '0_to_1_Ry', '1_to_4_Ry', '4_to_7_Ry', '7_to_10_Ry',
                           '10_to_13_Ry', '13_to_16_Ry', '16_to_19_Ry', '19_to_22_Ry', 'log10(Z/Z_sun)',
                           'log10(CF) [erg cm^{3} s^{-1}]', 'log10(HF) [erg cm^{3} s^{-1}]'
                           ]].explode(['log10(Z/Z_sun)', 'log10(CF) [erg cm^{3} s^{-1}]', 'log10(HF) [erg cm^{3} s^{-1}]'], ignore_index = True)
        # Convert the expanded columns to float (rather than 'object' datatype), needed for XGBoost to handle the data table
        data_df['log10(Z/Z_sun)'] = data_df['log10(Z/Z_sun)'].astype(float)
        data_df['log10(CF) [erg cm^{3} s^{-1}]'] = data_df['log10(CF) [erg cm^{3} s^{-1}]'].astype(float)
        data_df['log10(HF) [erg cm^{3} s^{-1}]'] = data_df['log10(HF) [erg cm^{3} s^{-1}]'].astype(float)

    # Make the target column
    data_df['target'] = data_df[target]
    
    return data_df
