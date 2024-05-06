# Module to handle reading in needed parameters from a configuration file

# Import configuration file parser
import configparser
# Import numpy for math
import numpy as np
# Import warnings                                                                                                                                               
import warnings
# Import os module to check for a file                                                                                                                           
import os


def read_config_file(model_dir):
    '''
    Function to read in a config file for a given model and return parameters needed elsewhere.
    Input:
    model_dir (str): Path to the directory containing the configuration file (should be named 'config.ini').
    Output:
    config_entries (dict): A dictionary containing:
    -data_path (str): Path to directory containing the training data.
    -target (str): String labelling the target for XGBoost to predict.                                                                            
    -output (str): String specifying whether to read in CF or HF from data table.                                                                     
    -Z_vals (list): List of Z/Z_sun values to read in from data table.
    -rf_feats (str): Type of RF averaging to use ('bins' or 'rates')
    -scale_with_1_4_ry (bool): Whether or not to scale RF bins by the 1-4 Ry bin instead of 0.5-1 Ry (defaults to False)
    -random_seed (int): Random seed (for reproducibility)
    -train_frac (flt): Fraction of data to use for training (90%), rest for testing the trained model.
    -do_hp_val (bool): Flag determing whether or not to perform hyperparameter validation on 10% of the training data. 
    -features (list): List of parameters to use as features in the XGBoost model.
    -hp_search_space (dict): Dictionary containing hyperparameters and ranges to search over in Bayesian hyperparameter optimization.
    '''

    # Set up configuration file parsing
    config = configparser.ConfigParser()
    # Read in the configuration file (named 'config.ini').
    config.read(model_dir + '/config.ini')

    # Set up a dictionary to store all these entries
    config_entries = {}
    # Get the path to the training data
    config_entries['data_path'] = config['IO']['training_data_path']
    # Get string to label target column with
    config_entries['target'] = 'log10(' + config['IO']['output'] + ') [erg cm^{3} s^{-1}]'
    # Get string of target type (CF or HF) to read from training data
    config_entries['output'] = config['IO']['output']
    # Check if Z value is specified in config file
    if 'Z/Z_sun' in config['IO']:
        # If so, get the Z/Z_sun value
        Z = float(config['IO']['Z/Z_sun'])
        # Now check if it's equal to any of the allowed Z values
        for allowed_Z in [0, 0.1, 0.3, 1, 3]: 
            if Z == allowed_Z:
                # If so, only read in that fixed Z value
                config_entries['Z_vals'] = [allowed_Z]
    else: # If not, read in all Z values in the training data table
        config_entries['Z_vals'] = [0, 0.1, 0.3, 1, 3]
    # Read in the categorical value for what RF feature type to use
    config_entries['rf_feats'] = config['IO']['rf_feats']
    # Set scale_with_1_4_ry (Note: this is only used if rf_feats = bins)
    if 'scale_with_1_4_ry' in config['IO']:
        # If so, set the flag to True
        config_entries['scale_with_1_4_ry'] = True
        # Otherwise, set it to False
    else:
        config_entries['scale_with_1_4_ry'] = False        
        
    # Get random seed as an integer
    config_entries['random_seed'] = int(config['ml_data_prep']['random_seed'])
    # Get fraction of input data table to use for model training (rest for grid search validation).
    config_entries['train_frac'] = float(config['ml_data_prep']['train_frac'])
    # Get boolean flag for whether or not to perform hyperparameter validation
    config_entries['do_hp_val'] = config['ml_data_prep'].getboolean('do_hp_val')
    # If this flag is set to False, make sure there's a corresponding optimized hyperparameters file
    # If not, throw a warning
    if config_entries['do_hp_val'] == False:
        if not os.path.exists(model_dir + '/hp_val_best_params.txt'):
            # If not, issue a warning
            warnings.warn('No optimized hyperparameters found in model directory.  Please set do_hp_val = True in config file.')
    
    # Get list of features for the XGBoost model
    config_entries['features'] = [key + '_feat' for key in config['features']]

    # Set up dictionary for hyperparameter search space dimensions
    hp_search_space = {}
    # Loop through names of hyperparameters in the grid search only if flag to do hyperparameter validation is set to true
    if config_entries['do_hp_val'] == True:
        for key in config['hp_search_space']:
            hp_search_space[key] = [float(x.strip()) for x in config['hp_search_space'][key].split(',')]
        # Put this dictionary into config_entries
        config_entries['hp_search_space'] = hp_search_space
        
    return config_entries
