# Module to train and save an XGBoost model on training data

# Import xgboost
import xgboost as xgb 

def train_model(dtrain, hyperparams, model_dir):
    '''
    Function to train and save an xgboost model on training data, with given hyperparameters.
    Input:
    dtrain (DMatrix): DMatrix containing features, target for randomly selected subset of training data.
    hyperparams (dict): Dictionary of desired hyperparameter values for training model.
    model_dir (str): Path to directory containing relevant config file, to save the model.
    Output:
    model (XGBoost model): The trained model (also saved).
    '''
    
    # Train the model
    if 'n_estimators' in hyperparams: # If hyperparams contains a number of estimators, use it
        model = xgb.train(hyperparams, dtrain, hyperparams['n_estimators'])
    else: # Otherwise, don't
        model = xgb.train(hyperparams, dtrain)
    # Save the model
    model.save_model(model_dir + '/trained_model.txt')

    return model


    
