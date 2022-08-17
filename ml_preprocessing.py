# Module to create the data structures to feed into XGBoost

# Import pandas for dataframe handling
import pandas as pd
# Import train-test splitting from sklearn
from sklearn.model_selection import train_test_split
# Import xgboost
import xgboost as xgb

def get_split_xgboost_data(data_df, features, train_frac, random_seed):
    '''
    Function to create training, grid search validation, and test sets from the data, and package them as needed for XGBoost.
    Input:
    data_df (dataframe): The training data, with columns for the rescaled features, target.
    features (list of str): The list of desired XGBoost model features.
    train_frac (float): The fraction of the data to use for training.
    random_seed (int): Seed to make train-test split reproducible.
    Output:
    split_data (dict): A dictionary containing:
    -dtrain (DMatrix): DMatrix containing the features, labels for training data.
    -train_features (dataframe): Dataframe containing the features for training data.
    -train_labels (dataframe): Dataframe containing the labels for training data.
    -gs_features (dataframe): Dataframe containing the features for grid search data.
    -gs_labels (dataframe): Dataframe containing the labels for grid search data.
    -dtest (DMatrix): DMatrix containing the features, labels for test data.
    -test_features (dataframe): Dataframe containing the features for test data.
    -test_labels (dataframe): Dataframe containing the labels for test data.
    '''

    # Get dataframe containing just feature columns
    features_df = data_df[features]
    # get dataframe containing just target column
    target_df = data_df[['target']]

    # Do the train-test search split, using training fraction and random seed supplied
    train_features, test_features, train_labels, test_labels = train_test_split(features_df, target_df, 
                                                                            test_size = 1 - train_frac, 
                                                                            random_state = random_seed)
    # Now split off 10% of the training set for grid search/validation (change random seed by 1)
    train_features, gs_features, train_labels, gs_labels = train_test_split(train_features, train_labels,
                                                                            test_size = 0.1,
                                                                            random_state = random_seed + 1)
    
    # Convert the features, labels for train, test sets to XGBoost DMatrix datatype
    dtrain = xgb.DMatrix(train_features, train_labels)
    dtest = xgb.DMatrix(test_features, test_labels)

    # Package these all in a dictionary
    split_data = {'dtrain' : dtrain, 'train_features' : train_features, 'train_labels' : train_labels,
                  'gs_features' : gs_features, 'gs_labels' : gs_labels,
                  'dtest' : dtest, 'test_features' : test_features, 'test_labels' : test_labels}
    # Return the packaged dictionary
    return split_data
