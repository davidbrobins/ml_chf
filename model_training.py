# Module to train and save an XGBoost model on training data

# Import xgboost
import xgboost as xgb
# Import loading from pickle
from pickle import load
# Import warnings
import warnings
# Import os module to check for a file
import os

def train_model(train_features, train_labels, model_dir, opt_type):
    '''
    Function to train and save an xgboost model on training data, with optimized hyperparameters.
    Input:
    train_features (dataframe): Dataframe containing features for training set.
    train_labels (dataframe): Dataframe containg target for training set.
    model_dir (str): Path to directory containing relevant config file, to save the model.
    opt_type (str): Two-letter code specifying method used to find hyperparameters (currently: 'gs' for grid search, 'sp' for scipy optimization, 'bs' for Bayes search).
    Output:
    model (XGBoost model): The trained model (also saved).
    '''

    # Check if pickle file containing optimized hyperparameters exists
    if not os.path.exists(model_dir + '/' + opt_type + '_best_params.pkl'):
        # If not, issue a warning
        warnings.warn('No optimized hyperparameters found in model directors. Please run hyperparameter optimization (rungrid.py or runsciopt.py) before training model.')
    hyperparams = load(open(model_dir + '/' + opt_type + '_best_params.pkl', 'rb'))
    # Set up to run on gpu
    hyperparams['tree_method'] = 'gpu_hist'
    # Set up the model (with scikit learn API to save more about the model)
    model = xgb.XGBRegressor(**hyperparams, objective = 'reg:squarederror')
    # Train the model
    model.fit(train_features, train_labels)

    # Save the model
    model.save_model(model_dir + '/'+ opt_type + '_trained_model.txt')

    return model


    
