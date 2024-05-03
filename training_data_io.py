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

def get_training_data(data_path, rf_feats, target, output, Z_vals, scale_with_1_4_ry):
    '''
    Function to read in the needed susbset of training data, given relevant parameters from the config file.
    Input:
    data_path (str): Path to the directory containing the training data.
    rf_feats (str): Type of radiation field features to read in ('bins' or 'rates)
    target (str): The name of the target column
    output (str): Which of CF or HF is to be predicted.
    Z_vals (numpy array): Array containing the values of Z/Z_sun to read in
    scale_with_1_4_ry (bool): Whether or not to scale by 1-4 Ry (only used if rf_feats = 'bins')
    Output:
    data_df (dataframe): The desired subset of the training data.
    '''

    # Binned RF column names
    rf_bin_cols = ['log10(f_q)', 'log10(tau_0)', '0.5_to_1_Ry', '1_to_4_Ry', '4_to_7_Ry', 
                   '7_to_10_Ry', '10_to_13_Ry', '13_to_16_Ry', '16_to_19_Ry', '19_to_22_Ry'] 
    # Photoionization rate column names
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
    chf_cols = ['log10(n_H) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
                'log10(f_q)', 'log10(tau_0)', 'CF_Z_0','CF_Z_0.1', 'CF_Z_0.3', 'CF_Z_1', 'CF_Z_3',
                'HF_Z_0', 'HF_Z_0.1', 'HF_Z_0.3', 'HF_Z_1', 'HF_Z_3'] 
    # Create a blank dataframe to store the dataframes for each alpha
    alpha_dfs = {}
    # Define list of alpha values to loop through
    alpha_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Loop through alpha values
    for alpha in alpha_vals:
        # Check whether to read in binned RF data or photoionization rate data
        if rf_feats == 'bins':
            # Read in binned RF data at this alpha value
            rf_data = pd.read_csv(data_path + '/rf_bin_data/pratesLin3_'+str(alpha)+'.dat', sep= '\s+', names = rf_bin_cols)
        elif rf_feats == 'rates':
            # Read in photoionization rate data files at this alpha value
            p_rates = pd.read_csv(data_path + '/p_rate_data/prates_'+str(alpha)+'.dat', sep= '\s+', names = prate_cols)
            p_rates_MgFe = pd.read_csv(data_path + '/p_rate_data/pratesMgFe_' + str(alpha) + '.dat', sep = '\s+', names = prate_MgFe_cols)
            # Merge these two dataframes on the matching columns
            rf_data = p_rates.merge(p_rates_MgFe, on = ['log10(f_q)', 'log10(tau_0)', 'log10(Q_LW) [cm^3 s^{-1}]'])
        # Read in CHF data at this alpha value
        chf = pd.read_csv(data_path + '/raw_data/raw_'+str(alpha)+'.res', sep='\s+',
                          usecols = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20], names = chf_cols)
        # Merge the chf and p_rates_all dataframes on the matching columns (fq, tau0)
        merged = chf.merge(rf_data, on = ['log10(f_q)', 'log10(tau_0)'])
        # Scale RF averages by value containing J0
        # If using bins...
        if rf_feats == 'bins':
            # If scale_by_1_4_ry is True:
            if scale_by_1_4_ry == True:
                # Scale average RF in each bin by the value of 1-4 Ry bin
                for col in rf_bin_cols[4:]:
                    merged[col] = np.log10(merged[col]) - np.log10(merged['1_to_4_Ry'])
                merged['0.5_to_1_Ry'] = np.log10(merged['0.5_to_1_Ry']) - np.log10(merged['1_to_4_Ry'])
                # Scale average RF in 1-4 Ry bin by u0
                merged['1_to_4_Ry'] = np.log10(merged['1_to_4_Ry']) +  merged['log10(J_0/n_b/J_{MW})']
            # otherwise:
            else:
                # Scale average RF in each bin by the value of 0.5-1 Ry bin
                for col in rf_bin_cols[3:]:
                    merged[col] = np.log10(merged[col]) - np.log10(merged['0.5_to_1_Ry'])
                # Scale average RF in 0.5-1 Ry bin by u0
                merged['0.5_to_1_Ry'] = np.log10(merged['0.5_to_1_Ry']) +  merged['log10(J_0/n_b/J_{MW})']
        # If using rates...
        elif rf_feats == 'rates':
            # Scale each photoionization rate except Q_LW by Q_LW and take log10 (to match names)
            # (first two entries of prate_cols are NOT photoionization rates)
            for col in prate_cols[3:]:
                merged[col] = np.log10(merged[col]) - np.log10(merged['log10(Q_LW) [cm^3 s^{-1}]'])
            for col in prate_MgFe_cols[3:]:
                merged[col] = np.log10(merged[col]) - np.log10(merged['log10(Q_LW) [cm^3 s^{-1}]'])
            # Scale Q_LW by radiation field amplitude J_0 and take log10
            merged['log10(Q_LW) [cm^3 s^{-1}]'] = np.log10(merged['log10(Q_LW) [cm^3 s^{-1}]']) + merged['log10(J_0/n_b/J_{MW})']
        
        
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
