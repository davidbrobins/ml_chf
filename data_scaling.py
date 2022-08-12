# Module to create, train, and save data scalers for a given feature/target

# Import preprocessing module from sklearn
from sklearn import preprocessing
# Import pandas for dataframe handling
import pandas as pd
# Import module to go from feature names to training data columns
from column_definition import *
# Import saving method for pickle files
from pickle import dump

def rescale_feature(feature_name, data_df, model_dir, scaler_type = preprocessing.MinMaxScaler()):
    '''
    Function to create, train, and save a data scaler for a given feature.
    Input:
    feature_name (str): Name of feature to be rescaled.
    data_df (dataframe): The training data on which to train the scaler.
    model_dir (str): Path to the directory containing the relevant config file (saved scaler will be placed there).
    scaler_type (scaler): Type of scaler to apply (defaults ot MinMaxScaler())
    Output:
    data_df (dataframe): The training data, with a new column containing the rescaled feature.
    scaler: The trained scaler.
    '''
    
    # Fit the scaler
    scaler = scaler_type.fit(data_df[get_col_names(feature_name)].values.reshape(-1,1))
    # Apply the scaler, creating a new column in data_df labelled with the feature name
    data_df[feature_name] = scaler.transform(data_df[get_col_names(feature_name)].values.reshape(-1,1))
    # Save the scaler
    dump(scaler, open(model_dir + '/'+feature_name+'_scaler.pkl', 'wb'))
    
    # Return the updated dataframe
    return data_df, scaler

def rescale_target(target_name, data_df, model_dir, scaler_type = preprocessing.MinMaxScaler()):
    '''
    Function to create, train, and save a data scaler for a given target.
    Input:
    target_name (str): Name of target column to be rescaled.
    data_df (dataframe): The training data on which to train the scaler.
    model_dir (str): Path to the directory containing the relevant config file (saved scaler will be placed there).
    scaler_type (scaler): Type of scaler to apply (defaults ot MinMaxScaler())  
    Output:
    data_df (dataframe): The training data, with new column containing the rescaled target.
    scaler: The trained scaler.
    '''

    # Fit the scaler
    scaler = scaler_type.fit(data_df[target_name].values.reshape(-1,1))
    # Apply the scaler
    data_df['target'] = scaler.transform(data_df[target_name].values.reshape(-1,1))
    # Save the scaler
    dump(scaler, open(model_dir + '/target_scaler.pkl', 'wb'))

    # Return the updated dataframe
    return data_df, scaler


def rescale(features, target, data_df, model_dir, scaler_type = preprocessing.MinMaxScaler()):
    '''
    Function to create, train, and save data scalers for all given features, target.
    Input:
    features (list of str): Names of all features to rescale.
    target (str): Name of target to rescale.
    data_df (dataframe): The training data on which to train the scalers.
    model_dir (str): Path to the directory containing the relevant config file (saved scalers will be placed there).
    Output:
    data_df (dataframe): The training data, with new columns containing rescaled features, target.
    feature_scalers (dict): A dictionary of the trained feature scalers.
    target_scaler: The trained features scalers.
    '''

    # Create a blank dictionary for the feature scalers
    feature_scalers = {}
    # Iterate through features
    for feature in features:
        # Rescale that feature
        data_df, scaler = rescale_feature(feature, data_df, model_dir, scaler_type = scaler_type)
        # Put the feature into a dictionary
        feature_scalers[feature] = scaler
    # Rescale the target
    data_df, target_scaler = rescale_target(target, data_df, model_dir, scaler_type = scaler_type)

    # Return the updated dataframe
    return data_df, feature_scalers, target_scaler
