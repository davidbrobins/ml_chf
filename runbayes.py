# Python script to run hyperparameter Bayesian optimization search from a model directory with a config file
# Syntax to run: python runbayes.py configdir/
# (/ not needed)

# Imports:
# Command line argument handling
import sys
# Module to handle config files
import config
# Module to import training data
import training_data_io
# Module to handle scaling features/target
import data_scaling
# Module to package features/target for ML model, do train-test split
import ml_preprocessing
# Module to implement hyperparameter Bayesian search
import bayes_search_tools

# Unpack command line arguments (this file, path to config file directory)
(pyfilename, model_dir) = sys.argv

# Parse configuration files
config_entries = config.read_config_file(model_dir)

# Read in data
data_df = training_data_io.get_training_data(config_entries['data_path'], config_entries['alpha_vals'],
                                             config_entries['target'], config_entries['output'],
                                             config_entries['metallicity'], config_entries['restricted_params'])

# Apply feature and target scaling
data_df = data_scaling.rescale(config_entries['features'], config_entries['target'], config_entries['scale_chf'], data_df, model_dir)

# Do train-test split to get grid search data
split_data = ml_preprocessing.get_split_xgboost_data(data_df, config_entries['features'],
                                                     config_entries['train_frac'], config_entries['random_seed'])

# Do grid search
best_params = bayes_search_tools.do_bayes_search(split_data['gs_features'], split_data['gs_labels'],
                                                 config_entries['grid_search_params'], model_dir)
# Display the optimal hyperparameters from the grid search
print('Best parameters from grid search: \n', best_params)
