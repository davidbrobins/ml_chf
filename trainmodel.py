# Python script to train model on training data using optimized hyperparameter, then evaluate it on test set.
# Syntax to run: python trainmodel.py configdir/ opt_type
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
# Module to train model
import model_training
# Moduel to evaluate model
import model_evaluation

# Unpack command line arguments (this file, path to config file directory, two-letter code for the method used to find best model hyperparameters)
(pyfilename, model_dir, opt_type) = sys.argv

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

# Train the model
model = model_training.train_model(split_data['train_features'], split_data['train_labels'], model_dir, opt_type)

# Evaluate the model on test data
model_results = model_evaluation.evaluate_model(split_data['test_features'],
                                                split_data['test_labels'], model,
                                                config_entries['features'], config_entries['scale_chf'],
                                                model_dir, opt_type)


