# Module to set up and run a Bayesian optimization hyperparameter search and save the results

# Import scikit-optimize's Bayesian cross validation hyperparameter optimization
from skopt import BayesSearchCV
# Import xgboost
import xgboost as xgb
# Import saving method for pickle files
from pickle import dump
# Time module for timing the grid search
import time

def do_bayes_search(gs_features, gs_labels, grid_search_params, model_dir):
    '''
    Function to execute a Bayesian search over the validation data (from train-test split on the entire training data).
    Save grid search results as a dataframe and best parameters as text.
    Input:
    gs_features (dataframe): Dataframe containing features for grid search rows from train-test split.
    gs_labels (dataframe): Dataframe containing target for grid search rows from train-test split.
    grid_search_params (dict): Dictionary of hyperparameters to grid search through and values to consider, from config file.
    model_dir (str): Path to directory containing relevant config file, for saving results.
    Output:
    bs_best_params (dict): Dictionary of optimal values for each hyperparameter explored in the grid search.
    Saves pickle files of the grid search results and optimized hyperparameters.
    '''
    # Start timing
    start = time.time()
    
    # Set up the XGBoost model to optimize, which minimizes squared error
    regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

    # Make sure the regressor uses GPU
    grid_search_params['tree_method'] = ['gpu_hist']
    
    # Set up the grid search
    bayes_search = BayesSearchCV(estimator = regressor, # The model to optimize
                               param_grid = grid_search_params, # The parameter grid
                               scoring = 'neg_mean_squared_error', # The scoring system
                               # verbose = 2, # Display a lot of the output to track progress
                               n_jobs = -1) # Use all available CPU processors

    # Execute the grid search
    bayes_search.fit(gs_features, gs_labels)

    # Print how long it took
    print('Time for grid search: ', time.time()-start)

    # Get best parameters
    best_params = bayes_search.best_params_
    # Save them, marked as from grid search
    dump(best_params, open(model_dir + '/bs_best_params.pkl', 'wb'))
    # Return them
    return best_params
    
    
