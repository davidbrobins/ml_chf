# Module to create, train, and save data scalers for a given feature/target

# Import preprocessing module from sklearn
from sklearn import preprocessing
# Import pandas for dataframe handling
import pandas as pd
# Import module to go from feature names to training data columns
from column_definition import *
# Import saving method for pickle files
from pickle import dump
# Import deepcopy so that we can train new scalers without changing past ones
from copy import deepcopy

def rescale_feature(feature_name, data_df, scaler_type = preprocessing.MinMaxScaler()):
    '''
    Function to create, train, and save a data scaler for a given feature.
    Input:
    feature_name (str): Name of feature to be rescaled.
    data_df (dataframe): The training data on which to train the scaler.
    scaler_type (scaler): Type of scaler to apply (defaults ot MinMaxScaler())
    Output:
    data_df (dataframe): The training data, with a new column containing the rescaled feature.
    scaler: The trained scaler.
    '''
    
    # Fit the scaler
    scaler = scaler_type.fit(data_df[get_col_names(feature_name)].values.reshape(-1,1))
    # Apply the scaler, creating a new column in data_df labelled with the feature name
    data_df[feature_name] = scaler.transform(data_df[get_col_names(feature_name)].values.reshape(-1,1))
        
    # Return the updated dataframe
    return data_df, scaler

def rescale(features, target, data_df, model_dir, scaler_type = preprocessing.MinMaxScaler()):
    '''
    Function to create, train, and save data scalers for all given features.
    Input:
    features (list of str): Names of all features to rescale.
    target (str): Name of target.
    data_df (dataframe): The training data on which to train the scalers.
    model_dir (str): Path to the directory containing the relevant config file (saved scalers will be placed there).
    Output:
    data_df (dataframe): The training data, with new columns containing rescaled features, target.
    scalers (dict): A dictionary of the trained feature scalers and target scaler.
    '''

    # Create a blank dictionary for the feature scalers and target scaler
    scalers = {}
    # Iterate through features
    for feature in features:
        # Rescale that feature
        data_df, scaler = rescale_feature(feature, data_df, scaler_type = scaler_type)
        # Put the feature scaler into a dictionary
        scalers[feature] = deepcopy(scaler)
    # No need to scale the target.  Just set 'target' column to the target
    data_df['target'] = data_df[target]
    
    # Save the scalers
    dump(scalers, open(model_dir + '/scalers.pkl', 'wb'))
    
    # Return the updated dataframe
    return data_df
