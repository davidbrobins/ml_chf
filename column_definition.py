# Module containing dictionary, dictionary mapping to go between feature names in config files, columns from data tables

# Set up a dictionary containing the data column names as keys, corresponding feature names as values
feat_to_col = {'t_feat': 'log10(T) [K]', 'n_b_feat': 'log10(n_b) [cm^{-3}]', 'p_lw_feat': 'log10(P_LW) [s^{-1}]',
               'p_hi_feat': 'log10(P_HI) [s^{-1}]', 'p_hei_feat': 'log10(P_HeI) [s^{-1}]', 'p_heii_feat': 'log10(P_HeII) [s^{-1}]', 
               'p_cvi_feat': 'log10(P_CVI) [s^{-1}]', 'p_al13_feat': 'log10(P_Al13) [s^{-1}]', 'p_fe26_feat': 'log10(P_Fe26) [s^{-1}]', 
               'p_ci_feat': 'log10(P_CI) [s^{-1}]', 'p_c04_feat': 'log10(P_C04) [s^{-1}]', 'p_c05_feat': 'log10(P_C05) [s^{-1}]', 
               'p_o06_feat': 'log10(P_O06) [s^{-1}]', 'p_o08_feat': 'log10(P_O08) [s^{-1}]', 'p_f09_feat': 'log10(P_F09) [s^{-1}]',
               'p_ne10_feat': 'log10(P_Ne10) [s^{-1}]', 'p_na11_feat': 'log10(P_Na11) [s^{-1}]', 'p_mg12_feat': 'log10(P_Mg12) [s^{-1}]', 
               'p_si14_feat': 'log10(P_Si14) [s^{-1}]', 'p_s16_feat': 'log10(P_S16) [s^{-1}]', 'p_ar18_feat': 'log10(P_Ar18) [s^{-1}]',
               'p_ca20_feat': 'log10(P_Ca20) [s^{-1}]'}

def get_col_names(feature):
    '''
    Function to get data table column names corresponding to desired XGBoost model features.
    Input:
    feature (str): feature name (obtained from config file)
    Output:
    column (str): corresponding column name
    '''
    
    # Use dictionary mapping to get column names
    column = feat_to_col[feature]
    
    return column


