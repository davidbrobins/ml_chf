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

    # Photoionization rate data column names
    prate_cols = ['log10(f_q)', 'log10(tau_0)', 'log10(Q_LW) [cm^3 s^{-1}]', 'log10(Q_HI/Q_LW)', 'log10(Q_HeI/Q_LW)', 
                  'log10(Q_HeII/Q_LW)', 'log10(Q_CVI/Q_LW)', 'log10(Q_Al13/Q_LW)', 'log10(Q_Fe26/Q_LW)', 
                  'log10(Q_CI/Q_LW)', 'log10(Q_C04/Q_LW)', 'log10(Q_C05/Q_LW)', 'log10(Q_O06/Q_LW)', 
                  'log10(Q_O08/Q_LW)', 'log10(Q_F09/Q_LW)', 'log10(Q_Ne10/Q_LW)', 'log10(Q_Na11/Q_LW)',
                  'log10(Q_Mg12/Q_LW)', 'log10(Q_Si14/Q_LW)', 'log10(Q_S16/Q_LW)', 'log10(Q_Ar18/Q_LW)', 
                  'log10(Q_Ca20/Q_LW)']
    # Additional photoionization rate column names
    prate_MgFe_cols = ['log10(f_q)', 'log10(tau_0)', 'log10(Q_LW) [cm^3 s^{-1}]', 'log10(Q_MgI/Q_LW)', 'log10(Q_MgII/Q_LW)', 
                   'log10(Q_MgIII/Q_LW)', 'log10(Q_MgIV/Q_LW)', 'log10(Q_MgV/Q_LW)', 'log10(Q_MgVI/Q_LW)', 
                   'log10(Q_MgVII/Q_LW)', 'log10(Q_MgVIII/Q_LW)', 'log10(Q_MgIX/Q_LW)', 'log10(Q_MgX/Q_LW)', 
                   'log10(Q_MgXI/Q_LW)', 'log10(Q_MgXII/Q_LW)', 'log10(Q_FeI/Q_LW)', 'log10(Q_FeII/Q_LW)', 
                   'log10(Q_FeIII/Q_LW)', 'log10(Q_FeIV/Q_LW)', 'log10(Q_FeV/Q_LW)', 'log10(Q_FeVI/Q_LW)', 
                   'log10(Q_FeVII/Q_LW)', 'log10(Q_FeVIII/Q_LW)', 'log10(Q_FeIX/Q_LW)', 'log10(Q_FeX/Q_LW)', 
                   'log10(Q_FeXI/Q_LW)', 'log10(Q_FeXII/Q_LW)', 'log10(Q_FeXIII/Q_LW)', 'log10(Q_FeXIV/Q_LW)',
                   'log10(Q_FeXV/Q_LW)', 'log10(Q_FeXVI/Q_LW)', 'log10(Q_FeXVII/Q_LW)', 'log10(Q_FeXVIII/Q_LW)', 
                   'log10(Q_FeXIX/Q_LW)', 'log10(Q_FeXX/Q_LW)', 'log10(Q_FeXXI/Q_LW)', 'log10(Q_FeXXII/Q_LW)', 
                   'log10(Q_FeXXIII/Q_LW)', 'log10(Q_FeXXIV/Q_LW)', 'log10(Q_FeXXV/Q_LW)', 'log10(Q_FeXXVI/Q_LW)']
    # CHF data column names
    chf_cols = ['log10(n_b) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
                'log10(f_q)', 'log10(tau_0)', 'CF_Z_0','CF_Z_0.1', 'CF_Z_0.3', 'CF_Z_1', 'CF_Z_3',
                'HF_Z_0', 'HF_Z_0.1', 'HF_Z_0.3', 'HF_Z_1', 'HF_Z_3'] 
    # Create a blank dataframe to store the dataframes for each alpha
    alpha_dfs = {}
    
    # Loop through alpha values
    for alpha in alpha_vals:
        # Read in photoionization rate data at this alpha value
        p_rates = pd.read_csv(data_path + '/p_rate_data/prates_'+str(alpha)+'.dat', sep= '\s+', names = prate_cols)
        p_rates_MgFe = pd.read_csv(data_path + '/p_rate_data/pratesMgFe_' + str(alpha) + '.dat', sep = '\s+', names = prates_MgFe_cols)
        # Merge these two dataframes on the matching columns
        p_rates_all = p_rates.merge(p_rates_MgFe, on = ['log10(f_q)', 'log10(tau_0)', 'log10(Q_LW) [cm^3 s^{-1}]'])
        
        # Read in CHF data at this alpha value
        chf = pd.read_csv(data_path + '/raw_data/raw_'+str(alpha)+'.res', sep='\s+',
                          usecols = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20], names = chf_cols)
        # Merge the chf and p_rates_all dataframes on the matching columns (fq, tau0)
        merged = chf.merge(p_rates_all, on = ['log10(f_q)', 'log10(tau_0)'])
        # Now scale each photoionization rate except Q_LW by Q_LW and take log10 (to match names)
        # (first two entries of prate_cols are NOT photoionization rates)
        for col in prate_cols[3:]:
            merged[col] = np.log10(merged[col]) - np.log10(merged['log10(Q_LW) [cm^3 s^{-1}]'])
        for col in prate_MgFe_cols[3:]:
            merged[col] = np.log10(merged[col]) - np.log10(merged['log10(Q_LW) [cm^3 s^{-1}]'])
        # Scale Q_LW by radiation field amplitude J_0 and take log10
        merged['log10(Q_LW) [cm^3 s^{-1}]'] = np.log10(merged['log10(P_LW) [cm^3 s^{-1}]']) + merged['log10(J_0/n_b/J_{MW})']
        
        # Add a column with the alpha value
        merged['alpha'] = alpha

        # Put the merged dataframe in the dictionary
        alpha_dfs[alpha] = merged

    # Merge dataframes for all the alpha values
    data_df = pd.concat([alpha_dfs[alpha] for alpha in alpha_vals])

    # Create columns containing arrays of log10(CF), log10(HF) at Z/Z_sun values in Z_vals
    # To get these, apply series -> list helper function on CF/HF(Z) columns in each row
    data_df['log10(CF) [erg cm^{3} s^{-1}]'] = np.log10(data_df[['CF_Z_' + str(val) for val in Z_vals]]).apply(series_to_list, axis = 1)
    data_df['log10(HF) [erg cm^{3} s^{-1}]'] = np.log10(data_df[['HF_Z_' + str(val) for val in Z_vals]]).apply(series_to_list, axis = 1)
    # Before taking log of Z values, convert Z/Z_sun = 0 to 1e-4 (the real values
    for index in range(len(Z_vals)):
        if Z_vals[index] == 0:
            Z_vals[index] = 1e-4
    # Create a column which contains the values of log(Z/Z_sun) used                                                                                  
    # Note: the Z=0 columns are really calculated as Z/Z_sun = 1e-4                                                                                   
    data_df['log10(Z/Z_sun)'] = pd.Series([np.log10(Z_vals) for x in range(len(data_df.index))])
    # Now, expand these 3 aligned lists, with all elements getting their own row (and only take the columns we'll need later)
    data_df = data_df[['log10(n_H) [cm^{-3}]', 'log10(T) [K]', 'log10(Q_LW) [cm^3 s^{-1}]', 'log10(Q_HI/Q_LW)', 'log10(Q_HeI/Q_LW)',
                       'log10(Q_HeII/Q_LW)', 'log10(Q_CVI/Q_LW)', 'log10(Q_Al13/Q_LW)', 'log10(Q_CI/Q_LW)', 'log10(Q_C04/Q_LW)',
                       'log10(Q_C05/Q_LW)', 'log10(Q_O06/Q_LW)', 'log10(Q_O08/Q_LW)', 'log10(Q_F09/Q_LW)', 'log10(Q_Ne10/Q_LW)',
                       'log10(Q_Na11/Q_LW)', 'log10(Q_Si14/Q_LW)', 'log10(Q_S16/Q_LW)', 'log10(Q_Ar18/Q_LW)', 'log10(Q_Ca20/Q_LW)',
                       'log10(Q_MgI/Q_LW)', 'log10(Q_MgII/Q_LW)', 'log10(Q_MgIII/Q_LW)', 'log10(Q_MgIV/Q_LW)', 'log10(Q_MgV/Q_LW)',
                       'log10(Q_MgVI/Q_LW)', 'log10(Q_MgVII/Q_LW)', 'log10(Q_MgVIII/Q_LW)', 'log10(Q_MgIX/Q_LW)', 'log10(Q_MgX/Q_LW)',
                       'log10(Q_MgXI/Q_LW)', 'log10(Q_MgXII/Q_LW)', 'log10(Q_FeI/Q_LW)', 'log10(Q_FeII/Q_LW)', 'log10(Q_FeIII/Q_LW)',
                       'log10(Q_FeIV/Q_LW)', 'log10(Q_FeV/Q_LW)', 'log10(Q_FeVI/Q_LW)', 'log10(Q_FeVII/Q_LW)', 'log10(Q_FeVIII/Q_LW)',
                       'log10(Q_FeIX/Q_LW)', 'log10(Q_FeX/Q_LW)', 'log10(Q_FeXI/Q_LW)', 'log10(Q_FeXII/Q_LW)', 'log10(Q_FeXIII/Q_LW)',
                       'log10(Q_FeXIV/Q_LW)', 'log10(Q_FeXV/Q_LW)', 'log10(Q_FeXVI/Q_LW)', 'log10(Q_FeXVII/Q_LW)', 'log10(Q_FeXVIII/Q_LW)',
                       'log10(Q_FeXIX/Q_LW)', 'log10(Q_FeXX/Q_LW)', 'log10(Q_FeXXI/Q_LW)', 'log10(Q_FeXXII/Q_LW)', 'log10(Q_FeXXIII/Q_LW)',
                       'log10(Q_FeXXIV/Q_LW)', 'log10(Q_FeXXV/Q_LW)', 'log10(Q_FeXXVI/Q_LW)', 
                       'log10(Z/Z_sun)', 'log10(CF) [erg cm^{3} s^{-1}]', 'log10(HF) [erg cm^{3} s^{-1}]'
                       ]].explode(['log10(Z/Z_sun)', 'log10(CF) [erg cm^{3} s^{-1}]', 'log10(HF) [erg cm^{3} s^{-1}]'], ignore_index = True)
    # Convert the expanded columns to float (rather than 'object' datatype), needed for XGBoost to handle the data table
    data_df['log10(Z/Z_sun)'] = data_df['log10(Z/Z_sun)'].astype(float)
    data_df['log10(CF) [erg cm^{3} s^{-1}]'] = data_df['log10(CF) [erg cm^{3} s^{-1}]'].astype(float)
    data_df['log10(HF) [erg cm^{3} s^{-1}]'] = data_df['log10(HF) [erg cm^{3} s^{-1}]'].astype(float)

    # Make the target column
    data_df['target'] = data_df[target]
    
    return data_df
