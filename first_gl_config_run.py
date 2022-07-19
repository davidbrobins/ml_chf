# Import configuration file parser
import configparser
# Other imports
import pandas as pd # Pandas for dataframe handling
import numpy as np # Numpy for math
import matplotlib.pyplot as plt # Matplotlib for plotting
plt.style.use('ggplot') # Set plotting style
import xgboost as xgb # XGBoost
from sklearn.model_selection import train_test_split # Train-test split from sklearn
from sklearn.preprocessing import MinMaxScaler # Feature/label scaling tool from sklearn (also available: StandardScaler, RobustScaler)
from sklearn.model_selection import GridSearchCV # Hyperparameter optimization tool

# Read in configuration file
config = configparser.ConfigParser()
config.read('models/first_gl_config_run.ini')

# Get list of alpha values to read in
if 'alpha' in config['restricted_input_params']:
    # If alpha is restricted in config file, get its value
    alpha_vals = [float(config['restricted_input_params']['alpha'])]
else:
    # Otherwise, just use all 7 possible values
    alpha_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
print('Alpha values used:', alpha_vals)

# Photoionization rate data column names
prate_cols = ['log10(f_q)', 'log10(tau_0)', 'P_LW', 'P_HI', 'P_HeI', 
              'P_HeII', 'P_CVI', 'P_Al13', 'P_Fe26', 'P_CI', 'P_C04',
              'P_C05', 'P_O06', 'P_O08', 'P_F09', 'P_Ne10', 'P_Na11',
              'P_Mg12', 'P_Si14', 'P_S16', 'P_Ar18', 'P_Ca20'] 
# CHF data column names
chf_cols = ['log10(n_b) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
            'log10(f_q)', 'log10(tau_0)', 'CF_0', 'CF_1', 'CF_2', 'HF_0', 
            'HF_1', 'HF_2'] 
# Create a blank array to store the dataframes for each alpha
alpha_dfs = {}

# Photoionization rate data column names
prate_cols = ['log10(f_q)', 'log10(tau_0)', 'P_LW', 'P_HI', 'P_HeI', 
              'P_HeII', 'P_CVI', 'P_Al13', 'P_Fe26', 'P_CI', 'P_C04',
              'P_C05', 'P_O06', 'P_O08', 'P_F09', 'P_Ne10', 'P_Na11',
              'P_Mg12', 'P_Si14', 'P_S16', 'P_Ar18', 'P_Ca20'] 
# CHF data column names
chf_cols = ['log10(n_b) [cm^{-3}]', 'log10(T) [K]', 'log10(J_0/n_b/J_{MW})', 
            'log10(f_q)', 'log10(tau_0)', 'CF_0', 'CF_1', 'CF_2', 'HF_0', 
            'HF_1', 'HF_2'] 
# Get string to label target column with
target = config['target_params']['output'] + '_Z_' + config['target_params']['Z']
# Get numerical value of metallicity
metallicity = int(config['target_params']['Z'])
# Create a blank array to store the dataframes for each alpha
alpha_dfs = {}
for alpha in alpha_vals:
    # Read in photoionization rate data at this alpha value
    p_rates = pd.read_csv('data/p_rate_data/prates_'+str(alpha)+'.dat', sep= '\s+', names = prate_cols)
    # Read in CHF data at this alpha value
    chf = pd.read_csv('data/chf_data/d'+str(alpha)+'.res', skiprows = 2, sep='\s+',
                      usecols = [0, 1, 2, 3, 4, 11, 12, 13, 16, 17, 18], names = chf_cols)
    # Calculate target
    chf[target] = sum([chf[config['target_params']['output']+'_'+str(i)]*(2**i) for i in range(metallicity+1)])
    # Merge the two dataframes on the matching columns (fq, tau0)
    merged = chf.merge(p_rates, on = ['log10(f_q)', 'log10(tau_0)'])
    # Now scale each photoionization rate by J0 (first two entries of prate_cols are NOT photoionization rates)
    for col in prate_cols[2:]:
        merged[col] = merged[col] * 10 ** merged['log10(J_0/n_b/J_{MW})']
    # Add a column with the alpha value
    merged['alpha'] = alpha
    # Restrict any input parameters specified in the config file
    for key in config['restricted_input_params']: # Check which parameters are restricted
        merged = merged.loc[(merged[key] == float(config['restricted_input_params'][key]))] # Restrict them
    # Put the merged dataframe in the array
    alpha_dfs[alpha] = merged

# Merge dataframes for all the alpha values
all_data = pd.concat([alpha_dfs[alpha] for alpha in alpha_vals])
print(all_data.iloc[0])

# Construct and apply scalers for all feature and target quantities
# target scaler
target_scaler = MinMaxScaler().fit(np.log10(all_data[target].values).reshape(-1,1))
all_data['target'] = target_scaler.transform(np.log10(all_data[target].values).reshape(-1,1))

# Define a function to go from feature name in config file to column name
def feature_to_column(feature_name):
    if feature_name == 'T' or feature_name == 't':
        return 'log10(T) [K]'
    elif feature_name == 'n_b':
        return 'log10(n_b) [cm^{-3}]'
    else:
        return feature_name.upper()

# Loop through all features (for now, I'll do this manually since there are some quirks)
T_scaler =  MinMaxScaler().fit(all_data['log10(T) [K]'].values.reshape(-1, 1)) # Fit the scaler
all_data['T_feat'] = T_scaler.transform(all_data['log10(T) [K]'].values.reshape(-1, 1)) # Apply the scaler
P_HI_scaler = MinMaxScaler().fit(np.log10(all_data['P_HI'].values).reshape(-1, 1)) # Fit the scaler
all_data['P_HI_feat'] = P_HI_scaler.transform(np.log10(all_data['P_HI'].values).reshape(-1, 1)) # Apply the scaler
P_HeI_scaler = MinMaxScaler().fit(np.log10(all_data['P_HeI'].values).reshape(-1, 1)) # Fit the scaler                                                                                             
all_data['P_HeI_feat'] = P_HeI_scaler.transform(np.log10(all_data['P_HeI'].values).reshape(-1, 1)) # Apply the scaler  
P_HeII_scaler = MinMaxScaler().fit(np.log10(all_data['P_HeI'].values).reshape(-1, 1)) # Fit the scaler                                                                                            
all_data['P_HeII_feat'] = P_HeII_scaler.transform(np.log10(all_data['P_HeII'].values).reshape(-1, 1)) # Apply the scaler  
P_CVI_scaler = MinMaxScaler().fit(np.log10(all_data['P_CVI'].values).reshape(-1, 1)) # Fit the scaler                                                                                             
all_data['P_CVI_feat'] = P_CVI_scaler.transform(np.log10(all_data['P_CVI'].values).reshape(-1, 1)) # Apply the scaler  

# Get list of features
features_list = ['T_feat', 'P_HI_feat', 'P_HeI_feat', 'P_HeII_feat', 'P_CVI_feat']

# Extract data columns of features, target
features = all_data[features_list]
labels = all_data[['target']]
print(features.iloc[0])
print(labels.iloc[0])

# Do a train-test split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.33, random_state = 24)

# Convert the train, test dataframes to XGBoost's DMatrix
dtrain = xgb.DMatrix(train_features, train_labels)
dtest = xgb.DMatrix(test_features, test_labels)

# Get dictionary of parameter values for the grid search from config file
grid_search_params = {}
for key in config['grid_search_params']:
    if key == 'max_depth' or key == 'lambda' or key == 'alpha' or key == 'n_estimators':
        grid_search_params[key] = [int(x.strip()) for x in config['grid_search_params'][key].split(',')]
    else:
        grid_search_params[key] = [float(x.strip()) for x in config['grid_search_params'][key].split(',')]
# Add a sampling method (not in config file)
grid_search_params['sampling_method'] = ['uniform']

print(grid_search_params)

# Set up the XGBoost model to optimize hyperparameters for
regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

# Set up the grid search
grid_search = GridSearchCV(estimator = regressor, # The model to optimize
                           param_grid = grid_search_params, # The parameter grid
                           scoring = 'neg_mean_squared_error', # The scoring system
                           verbose = 2 # How much info to print
                           )

# Execute the grid search
grid_search.fit(train_features, train_labels)

# Get best parameters
best_params = grid_search.best_params_
print('Hyperparameters from grid search: ', best_params)

# Train and save a model using those parameters
model = xgb.train(best_params, dtrain, best_params['n_estimators'])
model.save_model("models/first_model.txt")

# Get predictions from the model
ypred = model.predict(dtest)

# Now we need to inverse scale the target and features
test_features[target + '_pred'] = 10**(target_scaler.inverse_transform(ypred.reshape(-1,1)))
# Features (For now, I redefine the feature scalers, but hopefully this can be avoided in the more modular future)
test_features['log_T'] = T_scaler.inverse_transform(test_features['T_feat'].values.reshape(-1,1))
test_features['log_P_HI'] = P_HI_scaler.inverse_transform(test_features['P_HI_feat'].values.reshape(-1,1))
test_features['log_P_HeI'] = P_HeI_scaler.inverse_transform(test_features['P_HeI_feat'].values.reshape(-1,1))
test_features['log_P_HeII'] = P_HeII_scaler.inverse_transform(test_features['P_HeII_feat'].values.reshape(-1,1))
test_features['log_P_CVI'] = P_CVI_scaler.inverse_transform(test_features['P_CVI_feat'].values.reshape(-1,1))

# Save the model predictions
test_features.to_pickle('models/first_model_results.pkl')
# Save the true values to
test_labels.to_pickle('models/first_model_truth.pkl')
