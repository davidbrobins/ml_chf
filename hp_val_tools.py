# Module to set up and run a Bayesian optimization hyperparameter search and save the results

# Import scikit-optimize's Bayesian cross validation hyperparameter optimization
from skopt import BayesSearchCV
# Import methods to define space to earch
from skopt.space import Real, Categorical, Integer
# Import xgboost
import xgboost as xgb
# Import saving method for pickle files
from pickle import dump
# Time module for timing the grid search
import time

def do_bayes_search(hp_val_features, hp_val_labels, model_dir):
    '''
    Function to execute a Bayesian search over the validation data (from train-test split on the entire training data).
    Save grid search results as a dataframe and best parameters as text.
    Input:
    hp_val_features (dataframe): Dataframe containing features for hyperparameteer validation rows from train-test split.
    hp_val_labels (dataframe): Dataframe containing target for hyperparameter validation rows from train-test split.
    model_dir (str): Path to directory containing relevant config file, for saving results.
    Output:
    bs_best_params (dict): Dictionary of optimal values for each hyperparameter explored.
    Saves pickle files of the hyperparameter validation results and optimized hyperparameters.
    '''
    # Start timing
    start = time.time()
    
    # Set up the XGBoost model to optimize, which minimizes squared error
    regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

    # Set up the search space
    search_space = { 'max_depth' : Integer(4, 8, prior = 'uniform'), # Must be an integer, uniform prior between 4 and 8
                     'min_child_weight' : Real(0.1, 2, prior = 'log-uniform'), # Log-uniform prior between 0.1 and 2
                     'subsample' : Real(0.6, 1, prior = 'uniform'), # Uniform prior between 0.6 and 1
                     'colsample_bytree' : Real(0.6, 1, prior = 'uniform'), # Uniform prior between 0.6 and 1
                     'gamma' : Real(0, 1, prior = 'uniform'), # Uniform prior between 0 and 1
                     'eta' : Real(0.03, 0.3, prior = 'log-uniform'), # Log-uniform prior between 0.03 and 0.3
                     'n_estimators' : Integer(50, 500, prior = 'log-uniform'), # Must be an integer, log-uniform prior between 50 and 500
                     'tree_method' : Categorical(['gpu_hist']) # Ensure that model training in Bayesian search uses GPU
                     }
    print('Search space: ', search_space)
    
    # Set up the grid search
    bayes_search = BayesSearchCV(estimator = regressor, # The model to optimize
                                 search_spaces = search_space, # The parameter grid
                                 scoring = 'neg_mean_squared_error', # The scoring system
                                 # verbose = 2, # Display a lot of the output to track progress
                                 n_jobs = -1) # Use all available CPU processors

    # Execute the grid search
    bayes_search.fit(hp_val_features, hp_val_labels)

    # Print how long it took
    print('Time for Bayes search: ', time.time()-start)

    # Get best parameters
    best_params = bayes_search.best_params_
    # Save them, marked as from grid search
    dump(best_params, open(model_dir + '/hp_val_best_params.pkl', 'wb'))
    # Return them
    return best_params
    
    
