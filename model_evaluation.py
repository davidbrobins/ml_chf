# Module to get model predictions and unscaled features on the test set

# Import XGBoost
import xgboost as xgb
# Imort pandas for dataframe handling
import pandas as pd
# Import numpy
import numpy as np
# Import column names module
from column_definition import *
# From pickle import read
from pickle import load
# From pickle import saving
from pickle import dump

def evaluate_model(test_features, test_labels, model, features, model_dir):
    '''
    Function to run model predictions on test data.
    Input:
    test_features (dataframe): The test data features.
    test_labels (dataframe): The test data labels (truth).
    model (XGBoost model): The trained XGBoost model.
    features (list): List of feature names.
    model_dir (str): The path to the directory containing the appropriate config file (to save the results and read scalers).
    Output:
    model_results (dataframe): Unscaled features, target values (also saved).
    '''

    # Get model predictions
    pred = model.predict(test_features)

    # Save features, truth on test data for computing Shapley values
    dump(test_features, open(model_dir + '/test_features.pkl', 'wb'))
    dump(test_labels, open(model_dir + '/test_labels.pkl', 'wb'))
    
    # Define a new dataframe to hold the results.
    model_results = pd.DataFrame(index = range(len(test_features.index)), columns = [get_col_names(feature) for feature in features] + ['prediction', 'truth'])

    # Read in the feature and target scalers
    scalers = load(open(model_dir + '/scalers.pkl', 'rb'))
    
    # Unscale the features
    for feature in features: # Loop through all features
        # Get scaler
        scaler = scalers[feature]
        # Unscale the feature
        model_results[get_col_names(feature)] = scaler.inverse_transform(test_features[feature].values.reshape(-1,1))

    # Save the model predicted and true CHF values
    model_results['prediction'] = pred # Predictions are just the prediction (no unscaling required)
    model_results['truth'] = test_labels['target'].values # True values are just the test labels (no unscaling required)
    
    # Save the model results
    model_results.to_pickle(model_dir + '/model_results.pkl')

    return model_results

    
