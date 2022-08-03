# Module to handle reading in needed parameters from a configuration file

# Import configuration file parser
import configparser

def read_config_file(model_dir):
    '''
    Function to read in a config file for a given model and return parameters needed elsewhere.
    Input:
    model_dir (str): Path to the directory containing the configuration file (should be named 'config.ini').
    Output:
    data_path (str): Path to directory containing the training data.
    random_seed (int): Random seed (for reproducibility)
    alpha_vals (list): List of values of alpha parameter (which indexes training data) to use in training.
    target (str): String labelling the target for XGBoost to predict.
    output (str): String specifying whether model is for CF or HF.
    metallicity (int): Integer value (0, 1, or 2) at which output is evaluated (to get the target).
    restricted_params (dict): Dictionary containing training data parameters restricted to one value, and those values.
    features (list): List of parameters to use as features in the XGBoost model.
    grid_search_params (dict): Dictionary containing arrays of values for the hyperparameters to grid search through.
    '''

    # Set up configuration file parsing
    config = configparser.ConfigParser()
    # Read in the configuration file (named 'config.ini').
    config.read(model_dir + '/config.ini')

    # Get the path to the training data
    data_path = config['IO']['training_data_path']

    # Get random seed
    random_seed = config['IO']['random_seed']
    
    # Get list of alpha values to read in from training data
    # Check if alpha is given as a restricted parameter
    if 'alpha' in config['restricted_input_params']:
        # If so, get its value
        alpha_vals = [float(config['restricted_input_params']['alpha'])]
    else:
        # Otherwise, just use all 7 possible values
        alpha_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] 
    
    # Get string to label target column with
    target = 'log10('+ config['target_params']['output'] + '_Z_' + config['target_params']['Z'] + ') [erg cm^{3} s^{-1}]'
    # Get string of output type (CF or HF)
    output = config['target_params']['output']
    # Get numerical value of metallicity
    metallicity = int(config['target_params']['Z'])

    # Get list of features for the XGBoost model
    features = [key + '_feat' for key in config['features']]
    
    # Get dictionary of training data parameters to restrict, and the values to restrict them to
    # Set up dictionary
    restricted_params = {}
    # Loop through names of parameters to restrict
    for key in config['restricted_input_params']:
        # Set the key to the appropriate value, as a float
        restricted_params[key] = float(config['restricted_input_params'][key])

    # Get dictionary of hyperparameters to include in grid search, and the arrays of values to consider for them
    # Set up dictionary
    grid_search_params = {}
    # Loop through names of hyperparameters in the grid search
    for key in config['grid_search_params']:
        # Some hyperparameters need integer values
        if key == 'max_depth' or key == 'lambda' or key == 'alpha' or key == 'n_estimators':
            # Get list of hyperparameter values as integers
            grid_search_params[key] = [int(x.strip()) for x in config['grid_search_params'][key].split(',')]
        # Otherwise, float values are fine
        else:
            # Get list of hyperparameter values as floats
            grid_search_params[key] = [float(x.strip()) for x in config['grid_search_params'][key].split(',')]
    # Add a sampling method (not set in config file)
    grid_search_params['sampling_method'] = ['uniform']

    return data_path, random_seed, alpha_vals, target, output, metallicity, features, restricted_params, grid_search_params

