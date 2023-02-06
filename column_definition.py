# Module containing dictionary, dictionary mapping to go between feature names in config files, columns from data tables

# Set up a dictionary containing the data column names as keys, corresponding feature names as values
feat_to_col = {'t_feat': 'log10(T) [K]', 'n_h_feat': 'log10(n_H) [cm^{-3}]', 'z_feat': 'log10(Z/Z_sun)', 'q_lw_feat': 'log10(Q_LW) [cm^3 s^{-1}]',
               'q_hi_feat': 'log10(Q_HI/Q_LW)', 'q_hei_feat': 'log10(Q_HeI/Q_LW)', 'q_heii_feat': 'log10(Q_HeII/Q_LW)', 
               'q_cvi_feat': 'log10(Q_CVI/Q_LW)', 'q_al13_feat': 'log10(Q_Al13/Q_LW)', 'q_ci_feat': 'log10(Q_CI/Q_LW)',
               'q_c04_feat': 'log10(Q_C04/Q_LW)', 'q_c05_feat': 'log10(Q_C05/Q_LW)', 'q_o06_feat': 'log10(Q_O06/Q_LW)',
               'q_o08_feat': 'log10(Q_O08/Q_LW)', 'q_f09_feat': 'log10(Q_F09/Q_LW)', 'q_ne10_feat': 'log10(Q_Ne10/Q_LW)',
               'q_na11_feat': 'log10(Q_Na11/Q_LW)', 'q_si14_feat': 'log10(Q_Si14/Q_LW)', 'q_s16_feat': 'log10(Q_S16/Q_LW)',
               'q_ar18_feat': 'log10(Q_Ar18/Q_LW)', 'q_ca20_feat': 'log10(Q_Ca20/Q_LW)', 'q_mgi_feat': 'log10(Q_MgI/Q_LW)',
               'q_mgii_feat': 'log10(Q_MgII/Q_LW)', 'q_mgiii_feat': 'log10(Q_MgIII/Q_LW)', 'q_mgiv_feat': 'log10(Q_MgIV/Q_LW)',
               'q_mgv_feat': 'log10(Q_MgV/Q_LW)', 'q_mgvi_feat': 'log10(Q_MgVI/Q_LW)', 'q_mgvii_feat': 'log10(Q_MgVII/Q_LW)',
               'q_mgviii_feat': 'log10(Q_MgVIII/Q_LW)', 'q_mgix_feat': 'log10(Q_MgIX/Q_LW)', 'q_mgx_feat': 'log10(Q_MgX/Q_LW)',
               'q_mgxi_feat': 'log10(Q_MgXI/Q_LW)', 'q_mgxii_feat': 'log10(Q_MgXII/Q_LW)', 'q_fei_feat': 'log10(Q_FeI/Q_LW)',
               'q_feii_feat': 'log10(Q_FeII/Q_LW)', 'q_feiii_feat': 'log10(Q_FeIII/Q_LW)', 'q_feiv_feat': 'log10(Q_FeIV/Q_LW)',
               'q_fev_feat': 'log10(Q_FeV/Q_LW)', 'q_fevi_feat': 'log10(Q_FeVI/Q_LW)', 'q_fevii_feat': 'log10(Q_FeVII/Q_LW)',
               'q_feviii_feat': 'log10(Q_FeVIII/Q_LW)', 'q_feix_feat': 'log10(Q_FeIX/Q_LW)', 'q_fex_feat': 'log10(Q_FeX/Q_LW)',
               'q_fexi_feat': 'log10(Q_FeXI/Q_LW)', 'q_fexii_feat': 'log10(Q_FeXII/Q_LW)', 'q_fexiii_feat': 'log10(Q_FeXIII/Q_LW)',
               'q_fexiv_feat': 'log10(Q_FeXIV/Q_LW)', 'q_fexv_feat': 'log10(Q_FeXV/Q_LW)', 'q_fexvi_feat': 'log10(Q_FeXVI/Q_LW)',
               'q_fexvii_feat': 'log10(Q_FeXVII/Q_LW)', 'q_fexviii_feat': 'log10(Q_FeXVIII/Q_LW)', 'q_fexix_feat': 'log10(Q_FeXIX/Q_LW)',
               'q_fexx_feat': 'log10(Q_FeXX/Q_LW)', 'q_fexxi_feat': 'log10(Q_FeXXI/Q_LW)', 'q_fexxii_feat': 'log10(Q_FeXXII/Q_LW)',
               'q_fexxiii_feat': 'log10(Q_FeXXIII/Q_LW)', 'q_fexxiv_feat': 'log10(Q_FeXXIV/Q_LW)', 'q_fexxv_feat': 'log10(Q_FeXXV/Q_LW)',
               'q_fexxvi_feat': 'log10(Q_FeXXVI/Q_LW)'}

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


