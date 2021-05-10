import numpy as np
import pandas as pd
from learners import fit_predict_mse
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
from tqdm import tqdm
from utils.utils import *

# def tune_individually(train, n_iter=1000):
#   '''
#   Tune hyperparameters for each base learner individually
#   Inputs:
#     train: pd.DataFrame with training data
#     n_iter: number of hyperparameter settings to test for each base learner
#       default value is 1000
#   Returns:
#     rf_x: dict with best parameters for base learners for X learner
#     rf_t: dict with best parameters for base learners for T learner
#     rf_s: dict with best parameters for base learner for S learner
#   '''
#   start = time.time()

#   # Data preprocessing
#   X = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
#   y = train['Y']
#   W = train['treatment']

#   # Hyperparameter distributions for base learners
#   # X learner
#   params_X_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
#                  #'max_samples': np.arange(0.1, 1.05, 0.05),
#                  'max_features': range(1, len(X.columns)+1),
#                  'min_samples_leaf': range(1, 31)}
#   params_X_tau = {'max_samples': [0.1*(i+1) for i in range(10)],
#                   #'max_samples': np.arange(0.1, 1.05, 0.05),
#                   'max_features': range(1, len(X.columns)+1),
#                   'min_samples_leaf': range(1, 31)}
#   params_X_g = {'max_samples': [0.1*(i+1) for i in range(10)],
#                 #'max_samples': np.arange(0.1, 1.05, 0.05),
#                 'max_features': range(1, len(X.columns)+1),
#                 'min_samples_leaf': range(1, 31)}
#   # T learner
#   params_T_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
#                  'max_features': range(1, len(X.columns)+1),
#                  'min_samples_leaf': [1, 3, 5, 10, 30, 100]}
#   # S learner
#   params_S_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
#                  'max_features': range(1, len(X.columns)+1),
#                  'min_samples_leaf': [1, 3, 5, 10, 30, 100]}

#   # Do hyperparameter tuning for each base learner
#   mu0_base_X = RegressionForest(n_estimators=1000, honest=True, 
#                                 random_state=42, inference=False)
#   mu1_base_X = RegressionForest(n_estimators=1000, honest=True, 
#                                 random_state=42, inference=False)
#   tau0_base_X = RegressionForest(n_estimators=1000, honest=True, 
#                                  random_state=42, inference=False)
#   tau1_base_X = RegressionForest(n_estimators=1000, honest=True, 
#                                  random_state=42, inference=False)
#   g_base_X = RegressionForest(n_estimators=500, honest=True, 
#                               random_state=42, inference=False)
#   mu0_base_T = RegressionForest(n_estimators=500, honest=True, 
#                                 random_state=42, inference=False)
#   mu1_base_T = RegressionForest(n_estimators=500, honest=True, 
#                                 random_state=42, inference=False)
#   mu_base_S = RegressionForest(n_estimators=500, honest=True, 
#                                 random_state=42, inference=False)

#   search_mu0_X = RandomizedSearchCV(mu0_base_X, params_X_mu, n_iter=n_iter, n_jobs=4,
#                                     scoring='neg_mean_squared_error', random_state=42)  
#   search_mu1_X = RandomizedSearchCV(mu1_base_X, params_X_mu, n_iter=n_iter, n_jobs=4,
#                                     scoring='neg_mean_squared_error', random_state=42)
#   search_tau0_X = RandomizedSearchCV(tau0_base_X, params_X_tau, n_iter=n_iter, n_jobs=4,
#                                      scoring='neg_mean_squared_error', random_state=42)
#   search_tau1_X = RandomizedSearchCV(tau1_base_X, params_X_tau, n_iter=n_iter, n_jobs=4,
#                                      scoring='neg_mean_squared_error', random_state=42)
#   search_g_X = RandomizedSearchCV(g_base_X, params_X_g, n_iter=n_iter, n_jobs=4,
#                                   scoring='neg_mean_squared_error', random_state=42)
#   search_mu0_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, n_jobs=4,
#                                     scoring='neg_mean_squared_error', random_state=42)
#   search_mu1_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, n_jobs=4,
#                                     scoring='neg_mean_squared_error', random_state=42)
#   search_mu_S = RandomizedSearchCV(mu_base_S, params_S_mu, n_iter=n_iter, n_jobs=4,
#                                    scoring='neg_mean_squared_error', random_state=42)

#   tic = time.time()
#   search_mu0_X.fit(X[W==0], y[W==0])
#   print('trained mu0_base_X in ', str(time.time() - tic))
#   tic = time.time()
#   search_mu1_X.fit(X[W==1], y[W==1])
#   print('trained mu1_base_X in ', str(time.time() - tic))
#   tic = time.time()
#   search_g_X.fit(X, W)
#   print('trained g_X in ', str(time.time() - tic))
#   tic = time.time()
#   search_mu0_T.fit(X[W==0], y[W==0])
#   print('trained mu0_base_T in ', str(time.time() - tic))
#   tic = time.time()
#   search_mu1_T.fit(X[W==1], y[W==1])
#   print('trained mu1_base_T in ', str(time.time() - tic))
#   tic = time.time()
#   search_mu_S.fit(pd.concat([X, W], axis=1), y)
#   print('trained mu_base_S in ', str(time.time() - tic))

#   #Impute y0 for treated group using mu0
#   y0_treat = search_mu0_X.best_estimator_.predict(X[W==1]).flatten()
#   imputed_TE_treatment = y[W==1] - y0_treat

#   #Impute y1 for control group using mu1
#   y1_control = search_mu1_X.best_estimator_.predict(X[W==0]).flatten()
#   imputed_TE_control = y1_control - y[W==0]

#   # Fit tau0 and tau1 for X learner using best results from mu0 and mu1
#   tic = time.time()
#   search_tau0_X.fit(X[W==0], imputed_TE_control)
#   print('trained tau0_base_X in ', str(time.time() - tic))
#   tic = time.time()
#   search_tau1_X.fit(X[W==1], imputed_TE_treatment)
#   print('trained tau1_base_X in ', str(time.time() - tic))

#   rf_x = {'mu0': search_mu0_X.best_params_, 
#           'mu1': search_mu1_X.best_params_,
#           'tau0': search_tau0_X.best_params_,
#           'tau1': search_tau1_X.best_params_,
#           'g': search_g_X.best_params_}
#   rf_t = {'mu0': search_mu0_T.best_params_,
#           'mu1': search_mu1_T.best_params_}
#   rf_s = {'mu': search_mu_S.best_params_}
#   print('total execution time ', str(time.time() - start))
#   return (rf_x, rf_t, rf_s)

def tune_with_learners(train, val, sim, n_iter=1000):
  '''
  Tune hyperparameters for base learners by measuring performance in metalearners

  Inputs:
    train: pd.DataFrame with training data
    val: pd.DataFrame with validation data
    sim: simulation
    n_iter: number of hyperparameter settings to test for each metalearner
      default value is 1000
  Returns:
    rf_x: dict with best parameters for base learners for X learner
    rf_t: dict with best parameters for base learners for T learner
    rf_s: dict with best parameters for base learner for S learner
  '''
  start = time.time()
  X = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
  rng = np.random.default_rng(42)

  # Initialize meta_base_dict with all rf
  meta_base_dict = {'T': {'mu_0': 'rf', 'mu_1': 'rf', 'g': 'rfc'},
                    'S': {'mu': 'rf'},
                    'X': {'mu_0': 'rf', 'mu_1': 'rf', 'tau_0': 'rf', 'tau_1': 'rf', 'g': 'rfc'}}

  # Sample n_iter parameter settings for each base learner
  # X learner
  params_X_mu0 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                  'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                  'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                  'min_samples_leaf': rng.choice(range(1, 31), size=n_iter)}
  params_X_mu1 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                  'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                  'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                  'min_samples_leaf': rng.choice(range(1, 31), size=n_iter)}
  params_X_tau0 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                   'max_samples': rng.choice([0.1*(i+1) for i in range(9)], size=n_iter),
                   'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                   'min_samples_leaf': rng.choice(range(1, 31), size=n_iter)}
  params_X_tau1 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                   'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                   'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                   'min_samples_leaf': rng.choice(range(1, 31), size=n_iter)}
  params_X_g = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                   'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                   'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                   'min_samples_leaf': rng.choice(range(1, 31), size=n_iter)}                 
  # T learner
  params_T_mu0 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                  'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                  'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                  'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], size=n_iter)}
  params_T_mu1 = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                  'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                  'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                  'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], size=n_iter)}
  # S learner
  params_S_mu = {'n_estimators': rng.choice(np.arange(48, 496, 16, dtype=int), size=n_iter),
                 'max_samples': np.around(rng.choice([0.1*(i+1) for i in range(9)], size=n_iter), decimals=1),
                 'max_features': rng.choice(range(1, len(X.columns)+1), size=n_iter),
                 'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], size=n_iter)}

  # Find best configuration for each learner
  mse_list_X = []
  mse_list_T = []
  mse_list_S = []

  for i in tqdm(range(n_iter)):
    # X learner
    start_x = time.time()

    # Make empty X hyperparam dictionary to populate
    print('Tuning X learner')
    X_dict = {"mu_0": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}, 
              "mu_1": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}, 
              "tau_0": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}, 
              "tau_1": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}, 
              "g": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}}

    ith_mu0_items = {val: params_X_mu0[val][i] for val in params_X_mu0}
    print('mu0 params:', ith_mu0_items)
    X_dict['mu_0'][sim] = ith_mu0_items
    
    ith_mu1_items = {val: params_X_mu1[val][i] for val in params_X_mu1}
    print('mu1 params:', ith_mu1_items)
    X_dict['mu_1'][sim] = ith_mu1_items
    
    ith_tau0_items = {val: params_X_tau0[val][i] for val in params_X_tau0}
    print('tau0 params:', ith_tau0_items)
    X_dict['tau_0'][sim] = ith_tau0_items
    
    ith_tau1_items = {val: params_X_tau1[val][i] for val in params_X_tau1}
    print('tau1 params:', ith_tau1_items)
    X_dict['tau_1'][sim] = ith_tau1_items
    
    ith_g_items = {val: params_X_g[val][i] for val in params_X_g}
    print('g params:', ith_g_items)
    X_dict['g'][sim] = ith_g_items

    config = config_from_json(meta='X', sim=sim[-1], meta_base_dict=meta_base_dict, 
                                                hyperparams=X_dict)
    mse_X, _ = fit_predict_mse(train, val, config, export_preds=False)

    mse_list_X.append(mse_X)
    print('Finished fitting X learner in ', str(time.time()-start_x), ' s')


    # T learner
    start_t = time.time()
    # Make empty T hyperparam dictionary to populate
    print('Tuning T learner')
    T_dict = {"mu_0": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}, 
              "mu_1": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}},
              "g": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}}

    ith_mu0_items = {val: params_T_mu0[val][i] for val in params_T_mu0}
    print('mu0 params:', ith_mu0_items)
    T_dict['mu_0'][sim] = ith_mu0_items
    
    ith_mu1_items = {val: params_T_mu1[val][i] for val in params_T_mu1}
    print('mu1 params:', ith_mu1_items)
    T_dict['mu_1'][sim] = ith_mu1_items

    config = config_from_json(meta='T', sim=sim[-1], meta_base_dict=meta_base_dict, 
                                                hyperparams=T_dict)
    mse_T, _ = fit_predict_mse(train, val, config, export_preds=False)

    mse_list_T.append(mse_T)
    print('Finished fitting T learner in ', str(time.time()-start_t), ' s')


    # S learner
    start_s = time.time()
    # Make empty S hyperparam dictionary to populate
    print('Tuning S learner')
    S_dict = {"mu": {"simA": {}, "simB": {}, "simC": {}, "simD": {}, "simE": {}, "simF": {}}}

    ith_mu_items = {val: params_S_mu[val][i] for val in params_S_mu}
    print('mu params:', ith_mu_items)
    S_dict['mu'][sim] = ith_mu_items
    
    config = config_from_json(meta='S', sim=sim[-1], meta_base_dict=meta_base_dict, 
                                                hyperparams=S_dict)
    mse_S, _ = fit_predict_mse(train, val, config, export_preds=False)

    mse_list_S.append(mse_S)
    print('Finished fitting S learner in ', str(time.time()-start_s), ' s')

  # Find best params
  best_idx_X = np.argmax(np.asarray(mse_list_X))
  best_idx_T = np.argmax(np.asarray(mse_list_T))
  best_idx_S = np.argmax(np.asarray(mse_list_S))
  rf_x = {'mu0': {key : params_X_mu0[key][best_idx_pred_X]\
                  for key in params_X_mu0.keys()}, 
          'mu1': {key : params_X_mu1[key][best_idx_pred_X]\
                  for key in params_X_mu1.keys()}, 
          'tau0': {key : params_X_tau0[key][best_idx_pred_X]\
                   for key in params_X_tau0.keys()}, 
          'tau1': {key : params_X_tau1[key][best_idx_pred_X]\
                   for key in params_X_tau1.keys()}, 
          'g': {key : params_X_tau1[key][best_idx_pred_X]\
                   for key in params_X_tau1.keys()}}
  rf_t = {'mu0': {key : params_T_mu0[key][best_idx_T]\
                  for key in params_T_mu0.keys()}, 
          'mu1': {key : params_T_mu1[key][best_idx_T]\
                  for key in params_T_mu1.keys()},
          'g': {}}
  rf_s = {'mu': {key : params_S_mu[key][best_idx_S]\
                        for key in params_S_mu.keys()}}
  return (rf_x, rf_t, rf_s)

def main():
  # Set root directory
  base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

  # Read in data
  i = 0
  sims = ['simA', 'simB', 'simC', 'simD', 'simE', 'simF']
  train_size = 20000
  val_size = 10000
  rf_x_allsims = {}
  rf_t_allsims = {}
  rf_s_allsims = {}

  for sim in sims:
    samp_train_name = 'samp' + str(i+1) + '_train.parquet'
    full_train = pd.read_parquet(base_repo_dir / 'data' / sim / samp_train_name)
    train = strat_sample(full_train, train_size+val_size, replace=False, seed=42)
    val = strat_sample(train, val_size, replace=False, seed=42)
    train = train[~train.index.isin(val.index)]

    # Get best hyperparameters
    def np_encoder(object):
      if isinstance(object, np.generic):
          return object.item()
    # Uncomment next line and comment line after that to tune individually
    # rf_x, rf_t, rf_s = tune_individually(train, n_iter=1000)
    rf_x, rf_t, rf_s = tune_with_learners(train, val, sim, n_iter=2)
    print(rf_x, rf_t, rf_s)
    
    # add best params to allsims dictionaries
    for key, val in rf_x.items():
      if sim=='simA':
        rf_x_allsims[key]={}
        rf_x_allsims[key][sim] = val
      else:
        rf_x_allsims[key][sim] = val
    for key, val in rf_t.items():
      if sim=='simA':
        rf_t_allsims[key] = {}
        rf_t_allsims[key][sim] = val
      else:
        rf_t_allsims[key][sim] = val
    for key, val in rf_s.items():
      if sim=='simA':
        rf_s_allsims[key] = {}
        rf_s_allsims[key][sim] = val
      else:
        rf_s_allsims[key][sim] = val

  filepath_x = base_repo_dir / 'configurations/hyperparameters/rf_x_tuned.json'
  filepath_t = base_repo_dir / 'configurations/hyperparameters/rf_t_tuned.json'
  filepath_s = base_repo_dir / 'configurations/hyperparameters/rf_s_tuned.json'
  with open(filepath_x, 'w') as fp:
    json.dump(rf_x_allsims, fp, default=np_encoder)
  with open(filepath_t, 'w') as fp:
    json.dump(rf_t_allsims, fp, default=np_encoder)
  with open(filepath_s, 'w') as fp:
    json.dump(rf_s_allsims, fp, default=np_encoder)

  return

if __name__ == "__main__":
  main()
