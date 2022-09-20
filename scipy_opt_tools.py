# Module to run scipy optimization routines for a given validation set

# Imports:
# Module to get cross-validation fold scores
import cv_score
# Simplex optimization from scipy's optimization module
from scipy.optimize import minimize
# Numpy for math
import numpy as np
# Import saving method for picle files
from pickle import dump
# Import timing module
import time

def do_scipy_opt(gs_features, gs_labels, model_dir, params = np.array([6, 1.0, 1, 1, 0, 0.3, 100]), method = 'Nelder-Mead'):
    '''
    Function to run scipy optimization for hyperparameters on the given validation set, starting from given parameters.  
    Input:
    gs_features (dataframe): Dataframe containing features for validation rows from train-test split.
    gs_labels (dataframe): Dataframe containing target for validation rows from train-test split.
    model_dir: Path to directory containing relevant config file, for saving results.
    params (array): Array of real numbers which is converted to dictionary of hyperparameters as defined in cv_score module (defaults to default values of hyperparameters).  
    method (str): Name of the scipy optimization method to use (defaults to 'Nelder-Mead' for downhill simplex optimization).  
    Note that gradient and/or Hessian based methods cannot be used here.
    Output:
    best_params (dict): Dictionary of optimal values for each hyperparameters explored in the optimization.
    Saves pickle files of the optimized hyperparameters.
    '''
    print('Started from: \n', cv_score.array_to_hyperparams(params)) # Print starting hyperparameter values
    print('Optimization method: ', method)
    
    # Start time
    start = time.time()
    
    sp_opt = minimize(cv_score.get_model_cv_score, # The function to minimize (average mean squared error from 5 k-fold cross-validation)
                       params, # Starting values of hyperparameters (defaults to default values)
                       args = (gs_features, gs_labels), # The dataset to feed into the cross-validation (validation data)
                       method = method # Method for optimization
                       )

    # Extract best value of parameters, min error
    best_params = cv_score.array_to_hyperparams(sp_opt.x) # Convert to a dictionary with parameter names, allowable values

    # Print how long it took
    print('Time for scipy optimization: ', time.time()-start)
    
    # Save these results, marked as from scipy optimization
    dump(best_params, open(model_dir + '/sp_best_params.pkl', 'wb'))
    
    # Return the best parameters
    return best_params
