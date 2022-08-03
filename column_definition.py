# Module containing dictionary, dictionary mapping to go between feature names in config files, columns from data tables

# Set up a dictionary containing the data column names as keys, corresponding feature names as values
feat_to_col = {'t_feat': 'log10(T) [K]', 'n_b_feat': 'log10(n_b) [cm^{-3}]', 'p_lw_feat': 'P_LW',
               'p_hi_feat': 'P_HI', 'p_hei_feat': 'P_HeI', 'p_heii_feat': 'P_HeII', 'p_cvi_feat': 'P_CVI',
               'p_al13_feat': 'P_Al13', 'p_fe26_feat': 'P_Fe26', 'p_ci_feat': 'P_CI', 'p_c04_feat': 'P_C04',
               'p_c05_feat': 'P_C05', 'p_o06_feat': 'P_O06', 'p_o08_feat': 'P_O08', 'p_f09_feat': 'P_F09',
               'p_ne10_feat': 'P_Ne10', 'p_na11_feat': 'P_Na11', 'p_mg12_feat': 'P_Mg12', 
               'p_si14_feat': 'P_Si14', 'p_s16_feat': 'P_S16', 'p_ar18_feat': 'P_ar18',
               'p_ca20_feat': 'P_Ca20'}

def get_col_names(features):
    '''
    Function to get data table column names corresponding to desired XGBoost model features.
    Input:
    features (list of str): list of feature names (obtained from config file)
    Output:
    columns (list of str): list of corresponding column names
    '''
    
    # Use dictionary mapping to get column names
    columns = [feat_to_col[feat] for feat in features]
    
    return columns


