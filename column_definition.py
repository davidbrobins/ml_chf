# Module containing dictionary, dictionary mapping to go between feature names in config files, columns from data tables

# Set up a dictionary containing the data column names as keys, corresponding feature names as values
feat_to_col = {'t_feat': 'log10(T) [K]', 'n_b_feat': 'log10(n_b) [cm^{-3}]', 'p_lw_feat': 'log10(P_LW) [s^{-1}]',
               'p_hi_feat': 'log10(P_HI/P_LW)', 'p_hei_feat': 'log10(P_HeI/P_LW)', 'p_heii_feat': 'log10(P_HeII/P_LW)', 
               'p_cvi_feat': 'log10(P_CVI/P_LW)', 'p_al13_feat': 'log10(P_Al13/P_LW)', 'p_fe26_feat': 'log10(P_Fe26/P_LW)', 
               'p_ci_feat': 'log10(P_CI/P_LW)', 'p_c04_feat': 'log10(P_C04/P_LW)', 'p_c05_feat': 'log10(P_C05/P_LW)', 
               'p_o06_feat': 'log10(P_O06/P_LW)', 'p_o08_feat': 'log10(P_O08/P_LW)', 'p_f09_feat': 'log10(P_F09/P_LW)',
               'p_ne10_feat': 'log10(P_Ne10/P_LW)', 'p_na11_feat': 'log10(P_Na11/P_LW)', 'p_mg12_feat': 'log10(P_Mg12/P_LW)', 
               'p_si14_feat': 'log10(P_Si14/P_LW)', 'p_s16_feat': 'log10(P_S16/P_LW)', 'p_ar18_feat': 'log10(P_Ar18/P_LW)',
               'p_ca20_feat': 'log10(P_Ca20/P_LW)'}

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


