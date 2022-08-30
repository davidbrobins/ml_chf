# Module to set up and run a hyperparameter grid search and save the results

# Import sklearn's grid search
from sklearn.model_selection import GridSearchCV
# Import xgboost
import xgboost as xgb
# Import saving method for pickle files
from pickle import dump

def do_grid_search(gs_features, gs_labels, grid_search_params, model_dir):
    '''
    Function to execute a grid search over the data in the DMatrix dgs (from train-test split on the entire training data).
    Save grid search results as a dataframe and best parameters as text.
    Input:
    gs_features (dataframe): Dataframe containing features for grid search rows from train-test split.
    gs_labels (dataframe): Dataframe containing target for grid search rows from train-test split.
    grid_search_params (dict): Dictionary of hyperparameters to grid search through and values to consider, from config file.
    model_dir (str): Path to directory containing relevant config file, for saving results.
    Output:
    best_params (dict): Dictionary of optimal values for each hyperparameter explored in the grid search.
    Saves pickle files of the grid search results and optimized hyperparameters.
    '''

    # Set up the XGBoost model to optimize, which minimizes squared error
    regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

    # Set up the grid search
    grid_search = GridSearchCV(estimator = regressor, # The model to optimize
                               param_grid = grid_search_params, # The parameter grid
                               scoring = 'neg_mean_squared_error', # The scoring system
                               verbose = 2, # Display a lot of the output to track progress
                               n_jobs = -1) # Use all available CPU processors

    # Execute the grid search
    grid_search.fit(gs_features, gs_labels)

    # Get results, save them
    dump(grid_search.cv_results_, open(model_dir + '/grid_search_results.pkl', 'wb'))

    # Get best parameters
    best_params = grid_search.best_params_
    # Save them
    dump(best_params, open(model_dir + '/best_params.pkl', 'wb'))
    # Return them
    return best_params
    
    
