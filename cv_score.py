# Module to read in a validation set and return model score (averaged over 5 folds for cross-validation)

# Import statements
# XGBoost
import xgboost as xgb
# Cross-validation from sklearn's model selection library
from sklearn.model_selection import cross_val_score
# Numpy for math
import numpy as np

def array_to_hyperparams(params):
    '''
    Function to read in a numpy array of real numbers that will be converted to the parameters:
    max_depth, min_child_weight, subsample, colsample_bytree, gamma, eta, n_estimators (must be in that order).
    Input:
    params (array): An array of real numbers as described above.
    Output:
    param_dict (dict): A dictionary of the hyperparameters specified by the array.
    '''

    # Translate from the array to a dictionary with hyperparameter names
    param_dict = {'max_depth' : np.int(np.ceil(np.abs(params[0]))), # Must be a positive integer (so take ceiling of absolute value, force to be integer
                  'min_child_weight' : np.abs(params[1]), # Must be non-negative                                                                                                                     
                  'subsample' : params[2] if params[2] == 1.0 else params[2] - np.floor(params[2]), # Must be between 0 and 1                                                                           
                  'colsample_bytree' : params[3] if params[3] == 1.0 else params[3] - np.floor(params[3]), # Must be between 0 and 1                                                                 
                  'gamma' : np.abs(params[4]), # Must be non-negative                                                                                                                                
                  'eta' : np.abs(params[5]), # Must be non-negative                                                                                                                                  
                  'n_estimators' : np.int(np.ceil(np.abs(params[6]))), # Must be a positive integer
                  'tree_method' : 'gpu_hist' # Use GPU for training
                  }

    # Return the dictionary
    return param_dict

def get_model_cv_score(params, val_features, val_labels):
    '''
    Function to read in validation data and an arbitrary set of hyperparameters for the ML model and return a model score averaged over 5 cross-validations.
    Input:
    params (array): An array of real numbers that will be converted to the parameters:
    max_depth, min_child_weight, subsample, colsample_bytree, gamma, eta, n_estimators (must be in that order).
    val_features (dataframe): Dataframe containing the validation data features.
    val_labels (dataframe): Dataframe containing the validation data labels.
    Output:
    avg_score (float): The model score averaged over 5 cross-validations.
    '''
    
    # Set up an xgboost regression module using the given hyperparameters
    regressor = xgb.XGBRegressor(objective = 'reg:squarederror')
    
    # Set the parameters (convert using function defined above)
    regressor.set_params(**array_to_hyperparams(params))
    
    # Get scores for each 'fold' of the cross-validation (note: scores are NEGATIVE mean squared error)
    scores = cross_val_score(regressor, val_features, val_labels, n_jobs = -1, scoring = 'neg_mean_squared_error')

    # Return the average of the scores on each 'fold', times negative one to get the mean squared error (which we want to minimize)
    return -np.mean(scores)
    
