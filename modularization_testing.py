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

# Directory with config file
model_dir = 'models/modular_test/'

# Parse configuration file
config_entries = read_config_file(model_dir)
# Unpack the results
data_path = config_entries[0]
random_seed = config_entries[1]
train_frac = config_entries[2]
alpha_vals = config_entries[3]
target = config_entries[4]
output = config_entries[5]
metallicity = config_entries[6]
features = config_entries[7]
restricted_params = config_entries[8]
grid_search_params = config_entries[9]

print(data_path)
print(random_seed)
print(train_frac)
print(alpha_vals)
print(target)
print(output)
print(metallicity)
print(restricted_params)
print(features)
print(grid_search_params)

# Read in data
data_df = get_training_data(data_path, alpha_vals, target, output, metallicity, restricted_params)
print(data_df)

# Get column names corresponding to features
columns = [get_col_names(feat) for feat in features]
print(columns)

# Apply feature and target scaling
data_df = rescale(features, target, data_df, model_dir)
print(data_df[columns])
print(data_df[features])
print(data_df[target])
print(data_df['target'])

# Do train-grid search split and get DMatrixes for xgboost
dtrain, dgs = get_split_xgboost_data(data_df, features, train_frac, random_seed)
print(dtrain)
print(dgs)
