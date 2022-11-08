# Python script to run scipy hyperparameter optimization method from a model directory with a config file
# Syntax to run: python runsciopt.py configdir/ start_params_file
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
# Module to run simplex optimization
import scipy_opt_tools
# Numpy for math
import numpy as np
# Pickle to read in pickle file
from pickle import load

# Unpack command line arguments (this file, path to config file directory, filepath for file containing the parameters to start from)
(pyfilename, model_dir, params_file) = sys.argv
# Print the params_file file path
print('Path to initial hyperparameters: ', params_file)

# Parse configuration files
config_entries = config.read_config_file(model_dir)

# Parse the initial parameters
ihp = load(open(params_file, 'rb'))

# Read in data
data_df = training_data_io.get_training_data(config_entries['data_path'], config_entries['alpha_vals'],
                                             config_entries['target'], config_entries['output'],
                                             config_entries['restricted_params'])

# Apply feature and target scaling
data_df = data_scaling.rescale(config_entries['features'], config_entries['target'], config_entries['scale_chf'], data_df, model_dir)
print('Beginning train-test splitting')
# Do train-test split to get grid search data
split_data = ml_preprocessing.get_split_xgboost_data(data_df, config_entries['features'],
                                                     config_entries['train_frac'], config_entries['random_seed'])
print('Done train-test-splitting')

# Set up simplex optimizer
sp_best_params = scipy_opt_tools.do_scipy_opt(split_data['gs_features'], split_data['gs_labels'], model_dir,
                                              params = [ihp['max_depth'], ihp['min_child_weight'], ihp['subsample'], ihp['colsample_bytree'], ihp['gamma'], ihp['eta']*10, ihp['n_estimators']/100],
                                              method = 'Nelder-Mead') # Simplex optimization (can change for different optimization method

print('Best parameters from scipy optimizer: \n', sp_best_params)

