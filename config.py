# Module to handle reading in needed parameters from a configuration file

# Import configuration file parser
import configparser

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
    -metallicity (int): Integer value of metallicity (0, 1, or 2) at which to read in output from the data table.                                               
    -alpha_vals (list): List of values of alpha parameter (which indexes training data) to use in training.
    -restricted_params (dict): Dictionary containing training data parameters restricted to one value, and those values.
    -random_seed (int): Random seed (for reproducibility)
    -train_frac (flt): Fraction of data to use for training (90%) and grid search validation (10%), rest for testing the trained model.
    -scale_chf (bool): Whether or not to scale log10(CHF) [i.e. output] to interval 0-1 with min-max scaling
    -features (list): List of parameters to use as features in the XGBoost model.
    -grid_search_params (dict): Dictionary containing arrays of values for the hyperparameters to grid search through.
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
    config_entries['target'] = 'log10(' + config['IO']['output'] + '_Z_' + config['IO']['Z'] + ') [erg cm^{3} s^{-1}]'
    # Get string of target type (CF or HF) to read from training data
    config_entries['output'] = config['IO']['output']
    # Get integer value of metallicity to read from training data
    config_entries['metallicity'] = int(config['IO']['Z'])
    # Get list of alpha values to read in from training data
    if 'alpha' in config['IO']: # Check if value of alpha is specified
        # If so, get its value
        config_entries['alpha_vals'] = [float(config['IO']['alpha'])]
    else:
        # Otherwise, just use all 7 possible values                                                                                                            
        config_entries['alpha_vals'] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    # Get dictionary of training data parameters to restrict, and the values to restrict them to                                                            
    # Set up dictionary                                                                                                                                          
    restricted_params = {}
    # Loop through names of parameters to restrict                                                                                                          
    for key in config['IO']:
        # Check if key is a column which can be restricted
        if key == 'log10(T) [K]' or key == 'log10(n_b) [cm^{-3}]' or key == 'log10(J_0/n_b/J_{MW})' or key == 'log10(f_q)' or key == 'log10(tau_0)':
            # Set the key to the appropriate value, as a float                                                                                                   
            restricted_params[key] = float(config['IO'][key])
    config_entries['restricted_params'] = restricted_params
    # Get random seed as an integer
    config_entries['random_seed'] = int(config['ml_data_prep']['random_seed'])
    # Get fraction of input data table to use for model training (rest for grid search validation).
    config_entries['train_frac'] = float(config['ml_data_prep']['train_frac'])
    # Get boolean flag giving whether or not to scale output value to 0-1
    config_entries['scale_chf'] = eval(config['ml_data_prep']['scale_chf'])
    
    # Get list of features for the XGBoost model
    config_entries['features'] = [key + '_feat' for key in config['features']]
    
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
    config_entries['grid_search_params'] = grid_search_params
    
    return config_entries
