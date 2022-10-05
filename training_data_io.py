# Module handling reading in the needed subset of training data as a dataframe

import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math

def get_training_data(data_path, alpha_vals, target, output, metallicity, restricted_params):
    '''
    Function to read in the needed susbset of training data, given relevant parameters from the config file.
    Input:
    data_path (str): Path to the directory containing the training data.
    alpha_vals (list): Values of alpha parameter to include.
    target (str): The name of the target column
    output (str): Which of CF or HF is to be predicted.
    metallicity (int): Value of metallicity (0, 1, or 2 times solar) at which output is evaluated to get target.
    restricted_params (dict): Columns of training data to be restricted to a single value, and those values.
    Output:
    data_df (dataframe): The desired subset of the training data.
    '''

    # Photoionization rate data column names
    prate_cols = ['log10(f_q)', 'log10(tau_0)', 'log10(P_LW) [s^{-1}]', 'log10(P_HI/P_LW)', 'log10(P_HeI/P_LW)', 
                  'log10(P_HeII/P_LW)', 'log10(P_CVI/P_LW)', 'log10(P_Al13/P_LW)', 'log10(P_Fe26/P_LW)', 
                  'log10(P_CI/P_LW)', 'log10(P_C04/P_LW)', 'log10(P_C05/P_LW)', 'log10(P_O06/P_LW)', 
                  'log10(P_O08/P_LW)', 'log10(P_F09/P_LW)', 'log10(P_Ne10/P_LW)', 'log10(P_Na11/P_LW)',
                  'log10(P_Mg12/P_LW)', 'log10(P_Si14/P_LW)', 'log10(P_S16/P_LW)', 'log10(P_Ar18/P_LW)', 
                  'log10(P_Ca20/P_LW)'] 
    # CHF data column names
    chf_cols = ['log10(n_b) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
                'log10(f_q)', 'log10(tau_0)', 'CF_Z_0','CF_Z_0.1', 'CF_Z_0.3', 'CF_Z_1', 'CF_Z_3',
                'HF_Z_0', 'HF_Z_0.1', 'HF_Z_0.3', 'HF_Z_1', 'HF_Z_3'] 
    # Create a blank array to store the dataframes for each alpha
    alpha_dfs = {}
    
    # Loop through alpha values
    for alpha in alpha_vals:
        # Read in photoionization rate data at this alpha value
        p_rates = pd.read_csv(data_path + '/p_rate_data/prates_'+str(alpha)+'.dat', sep= '\s+', names = prate_cols)
        # Read in CHF data at this alpha value
        chf = pd.read_csv(data_path + '/raw_data/raw_'+str(alpha)+'.res', sep='\s+',
                          usecols = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20], names = chf_cols)
        
        # Calculate target
        chf[target] = (chf[output + '_Z_'+str(metallicity)]).transform(np.log10) # Take log, get value at right metallicity
        
        # Merge the two dataframes on the matching columns (fq, tau0)
        merged = chf.merge(p_rates, on = ['log10(f_q)', 'log10(tau_0)'])
        # Now scale each photoionization rate except P_LW by P_LW and take log10
        # (first two entries of prate_cols are NOT photoionization rates)
        for col in prate_cols[3:]:
            merged[col] = np.log10(merged[col]) - np.log10(merged['log10(P_LW) [s^{-1}]']) # Take difference (note that P_LW has not yet been scaled by radiation field amplitude, or had log10 taken
        # Scale P_LW by radiation field amplitude and take log10
        merged['log10(P_LW) [s^{-1}]'] = np.log10(merged['log10(P_LW) [s^{-1}]']) + merged['log10(J_0/n_b/J_{MW})']
        
        # Add a column with the alpha value
        merged['alpha'] = alpha

        # Restrict any input parameters specified in the config file
        # Loop through parameters to restrict
        for key in restricted_params: 
            # Restrict them to the specified values
            merged = merged.loc[(merged[key] == restricted_params[key])]

        # Put the merged dataframe in the dictionary
        alpha_dfs[alpha] = merged

    # Merge dataframes for all the alpha values
    data_df = pd.concat([alpha_dfs[alpha] for alpha in alpha_vals])
    
    return data_df
