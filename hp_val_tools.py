# Module to set up and run a Bayesian optimization hyperparameter search and save the results

# Import scikit-optimize's Bayesian cross validation hyperparameter optimization
from skopt import BayesSearchCV
# Import methods to define space to earch
from skopt.space import Real, Categorical, Integer
# Import convergence plot method
from skopt.plots import plot_convergence
# Import xgboost
import xgboost as xgb
# Import json to save optimal hyperparameters
import json
# Time module for timing the grid search
import time
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import seaborn for additional plotting
import seaborn as sns
# Numpy for math
import numpy as np
# Pnadas for dataframe handling
import pandas as pd
# Import saving pickle files
from pickle import dump

def do_bayes_search(hp_val_features, hp_val_labels, hp_search_space, model_dir):
    '''
    Function to execute a Bayesian search over the validation data (from train-test split on the entire training data).
    Save grid search results as a dataframe and best parameters as text.
    Input:
    hp_val_features (dataframe): Dataframe containing features for hyperparameteer validation rows from train-test split.
    hp_val_labels (dataframe): Dataframe containing target for hyperparameter validation rows from train-test split.
    hp_search_space (dict): Dictionary containing hyperparameter search space dimensions (read from config file).
    model_dir (str): Path to directory containing relevant config file, for saving results.
    Output:
    bs_best_params (dict): Dictionary of optimal values for each hyperparameter explored.
    Saves pickle files of the hyperparameter validation results and optimized hyperparameters.
    '''
    # Start timing
    start = time.time()
    
    # Set up the XGBoost model to optimize, which minimizes squared error
    regressor = xgb.XGBRegressor(objective = 'reg:squarederror')

    # Set up the search space
    search_space = { 'max_depth' : Integer(hp_search_space['max_depth'][0], hp_search_space['max_depth'][1], prior = 'uniform'), # Must be an integer, uniform prior
                     'min_child_weight' : Real(hp_search_space['min_child_weight'][0], hp_search_space['min_child_weight'][1], prior = 'log-uniform'), # Log-uniform prior
                     'subsample' : Real(hp_search_space['subsample'][0], hp_search_space['subsample'][1], prior = 'uniform'), # Uniform prior
                     'colsample_bytree' : Real(hp_search_space['colsample_bytree'][0], hp_search_space['colsample_bytree'][1], prior = 'uniform'), # Uniform prior
                     'gamma' : Real(hp_search_space['gamma'][0], hp_search_space['gamma'][1], prior = 'uniform'), # Uniform prior
                     'eta' : Real(hp_search_space['eta'][0], hp_search_space['eta'][1], prior = 'log-uniform'), # Log-uniform prior
                     'n_estimators' : Integer(hp_search_space['n_estimators'][0], hp_search_space['n_estimators'][1], prior = 'log-uniform'), # Must be an integer, log-uniform prior
                     'tree_method' : Categorical(['gpu_hist']) # Ensure that model training in Bayesian search uses GPU
                     }
    print('Search space: ', search_space)
    
    # Set up the grid search
    bayes_search = BayesSearchCV(estimator = regressor, # The model to optimize
                                 search_spaces = search_space, # The parameter grid
                                 scoring = 'neg_mean_squared_error', # The scoring system
                                 verbose = 1, # Display a lot of the output to track progress
                                 n_jobs = -1, # Use all available CPU processors
                                 n_points = 4) # Run 4 parameter sets in parallel

    # Callback handler                                                                                                                                                                  
    def on_step(optim_result):
        #print('Best score: ', bayes_search.best_score_)
        '''
        if time.time() - start > 1200: # If search has gone on more than 20 minutes                                                                                                     
            print("Interrupted after 20 minutes!")
            return True
        '''
    # Execute the grid search
    bayes_search.fit(hp_val_features, hp_val_labels, callback = on_step)

    # Print how long it took
    print('Time for Bayes search: ', time.time()-start)
    
    # Get best parameters
    best_params = bayes_search.best_params_
    # Save them
    with open(model_dir + '/hp_val_best_params.txt', 'w') as file:
        file.write(json.dumps(best_params))

    # Get convergence plot
    conv_plot = plot_convergence(bayes_search.optimizer_results_, yscale="log")
    # Save it
    plt.savefig(model_dir + '/hp_val_conv_test.pdf')

    # Get plot of parameter values vs. iteration (see below)
    hp_vs_iter = parameter_over_iterations(bayes_search)
    # Save it
    plt.savefig(model_dir + '/hp_vs_iter.pdf')

    # Save the optimizer results
    dump(bayes_search.optimizer_results_, open(model_dir + '/hp_val_results.pkl', 'wb'))
    
    # Return the optimal hyperparameters
    return best_params

def parameter_over_iterations(model_result):
  '''
  This function is generating a subplots with the hyperparameter values for each iteration and the overall performance score.
  The performance score is the difference between the best performing model and the worst performing model
  
  model_result: CV object

  Note: this function is borrowed from: https://towardsdatascience.com/improve-your-model-performance-with-bayesian-optimization-hyperparameter-tuning-4dbd7fe25b62
  '''
  param_list = list(model_result.cv_results_['params'][0].keys())
  max_col_plot = 2
  row_plot =int(np.ceil((len(param_list) + 1)/max_col_plot))
  fig, axs = plt.subplots(nrows=row_plot, ncols=np.min((max_col_plot, (len(param_list) + 1))), figsize=(30,12))
  for i, ax in enumerate(axs.flatten()):
    if i == len(param_list):
      break
    par = param_list[i]
    param_val = list()
    for par_dict in model_result.cv_results_['params']:
      param_val.append(par_dict[par])
    sns.barplot(y=param_val, x=np.arange(len(param_val)), ax=ax)
    ax.set_title(par)
  dt = pd.DataFrame({key:val for key,  val in model_result.cv_results_.items() if key.startswith('split')})
  mean_metric = dt.mean(axis=1)
  sns.barplot(y=(mean_metric.values + abs(np.min(mean_metric.values))), x=np.arange(len(mean_metric) ), ax=axs.flatten()[i])
  axs.flatten()[i].set_title('overall metric')
