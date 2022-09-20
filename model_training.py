# Module to train and save an XGBoost model on training data

# Import xgboost
import xgboost as xgb
# Import loading from pickle
from pickle import load
# Import warnings
import warnings
# Import os module to check for a file
import os

def train_model(dtrain, model_dir, opt_type):
    '''
    Function to train and save an xgboost model on training data, with optimized hyperparameters.
    Input:
    dtrain (DMatrix): DMatrix containing features, target for randomly selected subset of training data.
    model_dir (str): Path to directory containing relevant config file, to save the model.
    opt_type (str): Two-letter code specifying method used to find hyperparameters (currently: 'gs' for grid search, 'sp' for scipy optimization).
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
    
    # Train the model
    if 'n_estimators' in hyperparams: # If hyperparams contains a number of estimators, use it
        model = xgb.train(hyperparams, dtrain, hyperparams['n_estimators'])
    else: # Otherwise, don't
        model = xgb.train(hyperparams, dtrain)
    # Save the model
    model.save_model(model_dir + '/'+ opt_type + '_trained_model.txt')

    return model


    
