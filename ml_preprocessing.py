# Module to create the data structures to feed into XGBoost

# Import pandas for dataframe handling
import pandas as pd
# Import train-test splitting from sklearn
from sklearn.model_selection import train_test_split
# Import xgboost
import xgboost as xgb

def get_split_xgboost_data(data_df, features, train_frac, random_seed):
    '''
    Function to create train, test DMatrix to feed into XGBoost.
    Input:
    data_df (dataframe): The training data, with columns for the rescaled features, target.
    features (list of str): The list of desired XGBoost model features.
    train_frac (float): The fraction of the data to use for training.
    random_seed (int): Seed to make train-test split reproducible.
    Output:
    dtrain (DMatrix): DMatrix containing the features, labels for training data.
    dgs (DMatrix): DMatrix contianing the features, labels for grid search data.
    '''

    # Get dataframe containing just feature columns
    features_df = data_df[features]
    # get dataframe containing just target column
    target_df = data_df[['target']]

    # Do the train-grid search split, using training fraction and random seed supplied
    train_features, gs_features, train_labels, gs_labels = train_test_split(features_df, target_df, 
                                                                            test_size = 1 - train_frac, 
                                                                            random_state = random_seed)

    # Convert the features, labels to XGBoost DMatrix datatype
    dtrain = xgb.DMatrix(train_features, train_labels)
    dgs = xgb.DMatrix(gs_features, gs_labels)

    # Return the DMatrixes for training, grid search
    return dtrain, dgs
