# Module to run simplex optimization for a given validation set

# Imports:
# Module to get cross-validation fold scores
import cv_score
# Simplex optimization from scipy's optimization module
from scipy.optimize import fmin
# Numpy for math
import numpy as np
# Import saving method for picle files
from pickle import dump

# Set up simplex optimizer
so_best_params = fmin(cv_score.get_model_cv_score, np.array([5, 0.5, 0.6, 0.6, 0.5, 0.03, 20]),
                      args = (split_data['gs_features'], split_data['gs_labels']))

def do_simplex_opt(gs_features, gs_labels, params = np.array([6, 1.0, 1, 1, 0, 0.3, 100]), model_dir):
    '''
    Function to run simplex optimization for hyperparameters on the given validation set, starting from given parameters.  
    Input:
    gs_features (dataframe): Dataframe containing features for validation rows from train-test split.
    gs_labels (dataframe): Dataframe containing target for validation rows from train-test split.
    params (array): Array of real numbers which is converted to dictionary of hyperparameters as defined in cv_score module (defaults to default values of hyperparameters).
    model_dir (str): Path to directory containing relevant config file, for saving results.
    Output:
    best_params (dict): Dictionary of optimal values for each hyperparameters explored in the simplex optimization.
    Saves pickle files of the simplex optimization result (i.e. minimum mean squared error), optimized hyperparameters.
    '''

    sim_opt = fmin(cv_score.get_model_cv_score, # The function to minimize (average mean squared error from 5 k-fold cross-validation)
                   params, # Starting values of hyperparameters (defaults to default values)
                   args = (gs_features, gs_labels) # The dataset to feed into the cross-validation (validation data)
                   )

    # Extract best value of parameters, min error
    so_best_params = array_to_hyperparams(sim_opt[0]) # Convert to a dictionary with parameter names, allowable values
    so_min_error = sim_opt[1]

    # Save these results
    dump(so_best_params, open(model_dir + '/so_best_params.pkl', 'wb'))
    dump(so_min_error, open(model_dir + '/so_min_error.pkl', 'wb'))

    # Return the best parameters
    return so_best_params
