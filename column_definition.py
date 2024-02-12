# Module containing dictionary, dictionary mapping to go between feature names in config files, columns from data tables

# Set up a dictionary containing the data column names as keys, corresponding feature names as values
feat_to_col = {'t_feat': 'log10(T) [K]', 'n_h_feat': 'log10(n_H) [cm^{-3}]', 'z_feat': 'log10(Z/Z_sun)',
               '0_1_Ry_feat': '0_to_1_Ry', '1_4_Ry_feat': '1_to_4_Ry', '4_7_Ry_feat': '4_to_7_Ry',
               '7_10_Ry_feat': '7_to_10_Ry', '10_13_Ry_feat': '10_to_13_Ry', '13_16_Ry_feat': '13_to_16_Ry',
               '16_19_Ry_feat': '16_to_19_Ry', '19_22_Ry_feat': '19_to_22_Ry'}

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


