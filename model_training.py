# Module to train and save an XGBoost model on training data

# Import xgboost
import xgboost as xgb
# Import JSON to read in best hyperparameters (if needed)
import json

def train_model(train_features, train_labels, model_dir):
    '''
    Function to train and save an xgboost model on training data, with optimized hyperparameters.
    Input:
    train_features (dataframe): Dataframe containing features for training set.
    train_labels (dataframe): Dataframe containg target for training set.
    model_dir (str): Path to directory containing relevant config file, to save the model.
    Output:
    model (XGBoost model): The trained model (also saved).
    '''

    # Read in optimal hyperparameters
    with open(model_dir + 'hp_val_best_params.txt', 'r') as file:
        hyperparams = json.loads(file.read())
    # Set up to run on gpu
    hyperparams['tree_method'] = 'gpu_hist'
    # Set up the model (with scikit learn API to save more about the model)
    model = xgb.XGBRegressor(**hyperparams, objective = 'reg:squarederror')
    # Train the model
    model.fit(train_features, train_labels)

    # Save the model
    model.save_model(model_dir + '/trained_model.txt')

    return model


    
