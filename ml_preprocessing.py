# Module to create the data structures to feed into XGBoost

# Import pandas for dataframe handling
import pandas as pd
# Import train-test splitting from sklearn
from sklearn.model_selection import train_test_split
# Import xgboost
import xgboost as xgb

def get_split_xgboost_data(data_df, features, train_frac, random_seed, do_hp_val):
    '''
    Function to create training, grid search validation, and test sets from the data, and package them as needed for XGBoost.
    Input:
    data_df (dataframe): The training data, with columns for the rescaled features, target.
    features (list of str): The list of desired XGBoost model features.
    train_frac (float): The fraction of the data to use for training.
    random_seed (int): Seed to make train-test split reproducible.
    do_hp_val (bool): Boolean flag determining whether or not to perform hyperparameter validation on 10% of the training data.
    Output:
    split_data (dict): A dictionary containing:
    -train_features (dataframe): Dataframe containing the features for training data.
    -train_labels (dataframe): Dataframe containing the labels for training data.
    -hp_val_features (dataframe): Dataframe containing the features for hyperparameter data (only if do_hp_val = True).
    -hp_val_labels (dataframe): Dataframe containing the labels for hyperparameter validation data (only if do_hp_val = True).
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
    # If do_hp_val flag is set to true, perform another split, this time separating 10% of the test data
    # to use for hyperparameter validation
    if do_hp_val == True:
        train_features, hp_val_features, train_labels, hp_val_labels = train_test_split(train_features, train_labels,
                                                                                        test_size = 0.1,
                                                                                        random_state = random_seed + 1)

        # Package these all in a dictionary
        split_data = {'train_features' : train_features, 'train_labels' : train_labels,
                      'hp_val_features' : hp_val_features, 'hp_val_labels' : hp_val_labels,
                      'test_features' : test_features, 'test_labels' : test_labels}
    # Otherwise, just package the train and test dataframes into a dictionary
    else:
        split_data = {'train_features' : train_features, 'train_labels' : train_labels,
                      'test_features' : test_features, 'test_labels' : test_labels}
        
    # Return the packaged dictionary
    return split_data
