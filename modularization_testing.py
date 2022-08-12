# Working script to test code modularization

# Import module to handle config files
from config import *
# Import module to handle training data
from training_data_io import *
# Import module to convert feature names to column names
from column_definition import *
# Import module to rescale features, target
from data_scaling import *
# Import module to package features, target for xgboost
from ml_preprocessing import *
# Import module for hyperparameter grid search
from grid_search_tools import *
# Import module for model training
from model_training import *
# Import module for model evaluation on test set.
from model_evaluation import *

# Directory with config file
model_dir = 'models/modular_test/'

# Parse configuration file
config_entries = read_config_file(model_dir)
# Unpack the results
data_path = config_entries[0]
target = config_entries[1]
output = config_entries[2]
metallicity = config_entries[3]
alpha_vals = config_entries[4]
restricted_params = config_entries[5]
random_seed = config_entries[6]
train_frac = config_entries[7]
features = config_entries[8]
grid_search_params = config_entries[9]

print('Data path:', data_path)
print('Target name:', target)
print('CF or HF?:', output)
print('Metallicity:', metallicity)
print('Alpha values:', alpha_vals)
print('Restricted data table parameters:', restricted_params)
print('Random seed:', random_seed)
print('Fraction of data table to use for training/grid search:', train_frac)
print('Features list:', features)
print('Grid search parameters:', grid_search_params)

# Read in data
data_df = get_training_data(data_path, alpha_vals, target, output, metallicity, restricted_params)
print('All training data:', data_df)

# Get column names corresponding to features
columns = [get_col_names(feat) for feat in features]
print('Column names to use:', columns)

# Apply feature and target scaling
data_df, feature_scalers, target_scaler = rescale(features, target, data_df, model_dir)
print('Unscaled features:', data_df[columns])
print('Scaled features:', data_df[features])
print('Unscaled target:', data_df[target])
print('Scaled target:', data_df['target'])

# Do train-grid search split and get DMatrixes for xgboost
split_data = get_split_xgboost_data(data_df, features, train_frac, random_seed)
dtrain = split_data[0]
train_features = split_data[1]
train_labels = split_data[2]
gs_features = split_data[3]
gs_labels = split_data[4]
dtest = split_data[5]
test_features = split_data[6]
test_labels = split_data[7]
print('Training data features:', train_features)
print('Training data labels:', train_labels)
print('Grid search data features:', gs_features)
print('Grid search data labels:', gs_labels)
print('Testing data features:', test_features)
print('Testing data labels:', test_labels)

# Do grid search
best_params = do_grid_search(gs_features, gs_labels, grid_search_params, model_dir)
print('Best parameters from grid search:', best_params)

# Train model with optimized hyperparameters
model = train_model(dtrain, best_params, model_dir)

# Evaluate model on test set
model_results = evaluate_model(dtest, test_features, test_labels, model,
                               features, feature_scalers, target_scaler, model_dir)
print('Results on testing set:', model_results)
