# Module to get model predictions and unscaled features on the test set

# Import XGBoost
import xgboost as xgb
# Imort pandas for dataframe handling
import pandas as pd
# Import column names module
from column_definition import *

def evaluate_model(dtest, test_features, test_labels, model, features, feature_scalers, target_scaler, model_dir):
    '''
    Function to run model predictions on test data.
    Input:
    dtest (DMatrix): Test data features and labels packaged for XGBoost.
    test_features (dataframe): The test data features.
    test_labels (dataframe): The test data labels (truth).
    model (XGBoost model): The trained XGBoost model.
    features (list): List of feature names.
    feature_scalers (dict): Dictionary containing features as keys and corresponding
    data scalers as values.
    target_scaler (scaler): Scaler object for the target.
    model_dir (str): The path to the directory containing the appropriate config file (to save the results).
    Output:
    model_results (dataframe): Unscaled features, target values (also saved).
    '''

    # Get model predictions
    pred = model.predict(dtest)

    # Define a new dataframe to hold the results.
    model_results = pd.DataFrame(columns = [get_col_names(feature) for feature in features] + ['prediction', 'truth'])

    # Unscale the features
    for feature in features: # Loop through all features
        # Get scaler
        scaler = feature_scalers[feature]
        # Unscale the feature
        # model_results[get_col_names(feature)] =
        print(scaler.inverse_transform(test_features[feature].values.reshape(-1,1)))
    # Unscale the model prediction
    model_results['prediction'] = target_scaler.inverse_transform(pred.reshape(-1,1))
    # Unscale the true labels
    model_results['truth'] = target_scaler.inverse_transform(test_labels['target'].values.reshape(-1,1))

    # Save the model results
    model_results.to_pickle(model_dir + '/model_results.pkl')

    return model_results

    
