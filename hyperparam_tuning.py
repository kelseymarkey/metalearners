import numpy as np
import pandas as pd
from learners import fit_get_mse_s, fit_get_mse_t, fit_get_mse_x
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

def tune_individually(train, n_iter=1000):
  '''
  Tune hyperparameters for each base learner individually
  Inputs:
    train: pd.DataFrame with training data
    n_iter: number of hyperparameter settings to test for each base learner
      default value is 1000
  Returns:
    rf_x: dict with best parameters for base learners for X learner
    rf_t: dict with best parameters for base learners for T learner
    rf_s: dict with best parameters for base learner for S learner
  '''
  start = time.time()

  # Data preprocessing
  X = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
  y = train['Y']
  W = train['treatment']

  # Hyperparameter distributions for base learners
  # X learner
  params_X_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
                 #'max_samples': np.arange(0.1, 1.05, 0.05),
                 'max_features': range(1, len(X.columns)+1),
                 'min_samples_leaf': range(1, 31)}
  params_X_tau = {'max_samples': [0.1*(i+1) for i in range(10)],
                  #'max_samples': np.arange(0.1, 1.05, 0.05),
                  'max_features': range(1, len(X.columns)+1),
                  'min_samples_leaf': range(1, 31)}
  params_X_g = {'max_samples': [0.1*(i+1) for i in range(10)],
                #'max_samples': np.arange(0.1, 1.05, 0.05),
                'max_features': range(1, len(X.columns)+1),
                'min_samples_leaf': range(1, 31)}
  # T learner
  params_T_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
                 'max_features': range(1, len(X.columns)+1),
                 'min_samples_leaf': [1, 3, 5, 10, 30, 100]}
  # S learner
  params_S_mu = {'max_samples': [0.1*(i+1) for i in range(10)],
                 'max_features': range(1, len(X.columns)+1),
                 'min_samples_leaf': [1, 3, 5, 10, 30, 100]}

  # Do hyperparameter tuning for each base learner
  mu0_base_X = RegressionForest(n_estimators=1000, honest=True, 
                                random_state=42, inference=False)
  mu1_base_X = RegressionForest(n_estimators=1000, honest=True, 
                                random_state=42, inference=False)
  tau0_base_X = RegressionForest(n_estimators=1000, honest=True, 
                                 random_state=42, inference=False)
  tau1_base_X = RegressionForest(n_estimators=1000, honest=True, 
                                 random_state=42, inference=False)
  g_base_X = RegressionForest(n_estimators=500, honest=True, 
                              random_state=42, inference=False)
  mu0_base_T = RegressionForest(n_estimators=500, honest=True, 
                                random_state=42, inference=False)
  mu1_base_T = RegressionForest(n_estimators=500, honest=True, 
                                random_state=42, inference=False)
  mu_base_S = RegressionForest(n_estimators=500, honest=True, 
                                random_state=42, inference=False)

  search_mu0_X = RandomizedSearchCV(mu0_base_X, params_X_mu, n_iter=n_iter, n_jobs=-1,
                                    scoring='neg_mean_squared_error', random_state=42)  
  search_mu1_X = RandomizedSearchCV(mu1_base_X, params_X_mu, n_iter=n_iter, n_jobs=-1,
                                    scoring='neg_mean_squared_error', random_state=42)
  search_tau0_X = RandomizedSearchCV(tau0_base_X, params_X_tau, n_iter=n_iter, n_jobs=-1,
                                     scoring='neg_mean_squared_error', random_state=42)
  search_tau1_X = RandomizedSearchCV(tau1_base_X, params_X_tau, n_iter=n_iter, n_jobs=-1,
                                     scoring='neg_mean_squared_error', random_state=42)
  search_g_X = RandomizedSearchCV(g_base_X, params_X_g, n_iter=n_iter, n_jobs=-1,
                                  scoring='neg_mean_squared_error', random_state=42)
  search_mu0_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, n_jobs=-1,
                                    scoring='neg_mean_squared_error', random_state=42)
  search_mu1_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, n_jobs=-1,
                                    scoring='neg_mean_squared_error', random_state=42)
  search_mu_S = RandomizedSearchCV(mu_base_S, params_S_mu, n_iter=n_iter, n_jobs=-1,
                                   scoring='neg_mean_squared_error', random_state=42)

  tic = time.time()
  search_mu0_X.fit(X[W==0], y[W==0])
  print('trained mu0_base_X in ', str(time.time() - tic))
  tic = time.time()
  search_mu1_X.fit(X[W==1], y[W==1])
  print('trained mu1_base_X in ', str(time.time() - tic))
  tic = time.time()
  search_g_X.fit(X, W)
  print('trained g_X in ', str(time.time() - tic))
  tic = time.time()
  search_mu0_T.fit(X[W==0], y[W==0])
  print('trained mu0_base_T in ', str(time.time() - tic))
  tic = time.time()
  search_mu1_T.fit(X[W==1], y[W==1])
  print('trained mu1_base_T in ', str(time.time() - tic))
  tic = time.time()
  search_mu_S.fit(pd.concat([X, W], axis=1), y)
  print('trained mu_base_S in ', str(time.time() - tic))

  #Impute y0 for treated group using mu0
  y0_treat = search_mu0_X.best_estimator_.predict(X[W==1]).flatten()
  imputed_TE_treatment = y[W==1] - y0_treat

  #Impute y1 for control group using mu1
  y1_control = search_mu1_X.best_estimator_.predict(X[W==0]).flatten()
  imputed_TE_control = y1_control - y[W==0]

  # Fit tau0 and tau1 for X learner using best results from mu0 and mu1
  tic = time.time()
  search_tau0_X.fit(X[W==0], imputed_TE_control)
  print('trained tau0_base_X in ', str(time.time() - tic))
  tic = time.time()
  search_tau1_X.fit(X[W==1], imputed_TE_treatment)
  print('trained tau1_base_X in ', str(time.time() - tic))

  rf_x = {'mu0': search_mu0_X.best_params_, 
          'mu1': search_mu1_X.best_params_,
          'tau0': search_tau0_X.best_params_,
          'tau1': search_tau1_X.best_params_,
          'g': search_g_X.best_params_}
  rf_t = {'mu0': search_mu0_T.best_params_,
          'mu1': search_mu1_T.best_params_}
  rf_s = {'mu': search_mu_S.best_params_}
  print('total execution time ', str(time.time() - start))
  return (rf_x, rf_t, rf_s)

def tune_with_learners(train, test, n_iter=1000):
  '''
  Tune hyperparameters for base learners by measuring performance in metalearners

  Inputs:
    train: pd.DataFrame with training data
    test: pd.DataFrame with test data
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
  
  # Sample n_iter parameter settings for each base learner
  # X learner
  params_X_mu0 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                  'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                  'min_samples_leaf': rng.choice(range(1, 31), n_iter)}
  params_X_mu1 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                  'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                  'min_samples_leaf': rng.choice(range(1, 31), n_iter)}
  params_X_tau0 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                   'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                   'min_samples_leaf': rng.choice(range(1, 31), n_iter)}
  params_X_tau1 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                   'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                   'min_samples_leaf': rng.choice(range(1, 31), n_iter)}
  params_X_g = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                'min_samples_leaf': rng.choice(range(1, 31), n_iter)}
  rf_prop_X = rng.choice([True, False], n_iter)
  # T learner
  params_T_mu0 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                  'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                  'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], n_iter)}
  params_T_mu1 = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                  'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                  'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], n_iter)}
  # S learner
  params_S_mu = {'max_samples': rng.choice([0.1*(i+1) for i in range(10)], n_iter),
                 'max_features': rng.choice(range(1, len(X.columns)+1), n_iter),
                 'min_samples_leaf': rng.choice([1, 3, 5, 10, 30, 100], n_iter)}

  # Find best configuration for each learner
  mse_true_list_X = []
  mse_pred_list_X = []
  mse_list_T = []
  mse_list_S = []

  for i in range(n_iter):
    if i % 100 == 0:
      print(f'iteration {i+1} of {n_iter}')
    # X learner
    mu0_base_X = RegressionForest(n_estimators=1000, random_state=42, 
      min_samples_leaf=params_X_mu0['min_samples_leaf'][i], 
      max_features=params_X_mu0['max_features'][i], 
      max_samples=params_X_mu0['max_samples'][i],
      honest=True, inference=False)
    mu1_base_X = RegressionForest(n_estimators=1000, random_state=42, 
      min_samples_leaf=params_X_mu1['min_samples_leaf'][i], 
      max_features=params_X_mu1['max_features'][i], 
      max_samples=params_X_mu1['max_samples'][i],
      honest=True, inference=False)
    tau0_base_X = RegressionForest(n_estimators=1000, random_state=42, 
      min_samples_leaf=params_X_tau0['min_samples_leaf'][i], 
      max_features=params_X_tau0['max_features'][i], 
      max_samples=params_X_tau0['max_samples'][i],
      honest=True, inference=False)
    tau1_base_X = RegressionForest(n_estimators=1000, random_state=42, 
      min_samples_leaf=params_X_tau1['min_samples_leaf'][i], 
      max_features=params_X_tau1['max_features'][i], 
      max_samples=params_X_tau1['max_samples'][i],
      honest=True, inference=False)
    rf_prop = rf_prop_X[i]
    # g_base_X = RegressionForest(n_estimators=500, random_state=42, 
    #   min_samples_leaf=params_X_g['min_samples_leaf'][i], 
    #   max_features=params_X_g['max_features'][i], 
    #   max_samples=params_X_g['max_samples'][i],
    #   honest=True, inference=False)
    mse_true_X, mse_pred_X, _, _ = fit_get_mse_x(train, test, mu0_base_X, mu1_base_X,
      tau0_base_X, tau1_base_X, rf_prop=rf_prop)
    mse_true_list_X.append(mse_true_X)
    mse_pred_list_X.append(mse_pred_X)

    # T learner
    mu0_base_T = RegressionForest(n_estimators=500, random_state=42, 
      min_samples_leaf=params_T_mu0['min_samples_leaf'][i], 
      max_features=params_T_mu0['max_features'][i], 
      max_samples=params_T_mu0['max_samples'][i],
      honest=True, inference=False)
    mu1_base_T = RegressionForest(n_estimators=500, random_state=42, 
      min_samples_leaf=params_T_mu1['min_samples_leaf'][i], 
      max_features=params_T_mu1['max_features'][i], 
      max_samples=params_T_mu1['max_samples'][i],
      honest=True, inference=False)
    mse_T, _ = fit_get_mse_t(train, test, mu0_base_T, mu1_base_T)
    mse_list_T.append(mse_T)

    # S learner
    mu_base_S = RegressionForest(n_estimators=500, random_state=42, 
      min_samples_leaf=params_S_mu['min_samples_leaf'][i], 
      max_features=params_S_mu['max_features'][i], 
      max_samples=params_S_mu['max_samples'][i],
      honest=True, inference=False)
    mse_S, _ = fit_get_mse_s(train, test, mu_base_S)
    mse_list_S.append(mse_S)

  # Find best params
  best_idx_true_X = np.argmax(np.asarray(mse_true_X))
  best_idx_pred_X = np.argmax(np.asarray(mse_pred_X))
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
          'rf': rf_prop_X[best_idx_pred_X]}
  rf_t = {'mu0': {key : params_T_mu0[key][best_idx_T]\
                  for key in params_T_mu0.keys()}, 
          'mu1': {key : params_T_mu1[key][best_idx_T]\
                  for key in params_T_mu1.keys()}}
  rf_s = {'mu_base_S': {key : params_S_mu[key][best_idx_S]\
                        for key in params_S_mu.keys()}}
  return (rf_x, rf_t, rf_s)

def main():
  # Set root directory
  base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

  # Read in data
  train_size = 20000
  sim = 'simA'
  i = 0
  samp_train_name = 'samp' + str(i+1) + '_train.parquet'
  samp_test_name = 'samp' + str(i+1) + '_test.parquet'
  train = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_train_name)
  test = pd.read_parquet(base_repo_dir / 'data/' / str(train_size) / sim / samp_test_name)

  # Get best hyperparameters
  def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
  # rf_x, rf_t, rf_s = tune_individually(train, n_iter=1000)
  rf_x, rf_t, rf_s = tune_with_learners(train, test, n_iter=1000)
  print(rf_x, rf_t, rf_s)
  filepath_x = base_repo_dir + 'configurations/hyperparameters/rf_x.json'
  filepath_t = base_repo_dir + 'configurations/hyperparameters/rf_t.json'
  filepath_s = base_repo_dir + 'configurations/hyperparameters/rf_s.json'
  with open(filepath_x, 'w') as fp:
    json.dump(rf_x, fp, default=np_encoder)
  with open(filepath_t, 'w') as fp:
    json.dump(rf_t, fp, default=np_encoder)
  with open(filepath_s, 'w') as fp:
    json.dump(rf_s, fp, default=np_encoder)

  return

if __name__ == "__main__":
  main()
