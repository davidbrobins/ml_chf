# Python script to compare hyperparameter optimization methods from a model directory with a config file
# Syntax to run: python optimizers_test.py configdir/
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
# Module to implement hyperparameter grid search
import grid_search_tools
# Module to get cross-validation fold scores
import cv_score
# Module to run simplex optimization
import simplex_opt_tools
# Timing module
import time
# Numpy for math
import numpy as np

# Unpack command line arguments (this file, path to config file directory)
(pyfilename, model_dir) = sys.argv

# Parse configuration files
config_entries = config.read_config_file(model_dir)

# Read in data
data_df = training_data_io.get_training_data(config_entries['data_path'], config_entries['alpha_vals'],
                                             config_entries['target'], config_entries['output'],
                                             config_entries['restricted_params'])

# Apply feature and target scaling
data_df = data_scaling.rescale(config_entries['features'], config_entries['target'], config_entries['scale_chf'], data_df, model_dir)

# Do train-test split to get grid search data
split_data = ml_preprocessing.get_split_xgboost_data(data_df, config_entries['features'],
                                                     config_entries['train_frac'], config_entries['random_seed'])
# Get time before grid search
before_gs = time.time()

# Do grid search
gs_best_params = grid_search_tools.do_grid_search(split_data['gs_features'], split_data['gs_labels'],
                                               config_entries['grid_search_params'], model_dir)

# Display the optimal hyperparameters from the grid search
print('Best parameters from grid search: \n', gs_best_params)
print('Time for grid search:', time.time() - before_gs)

# Before simplex optimizer
before_so = time.time()

# Set up simplex optimizer
so_best_params = simplex_opt_tools.do_simplex_opt(split_data['gs_features'], split_data['gs_labels'],
                                                  model_dir)
print('Best parameters from simplex optimizer: \n', so_best_params)
print('Time for simplex optimization:', time.time() - before_so)
