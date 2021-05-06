import numpy as np
import pandas as pd
import learners
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import beta
import json
import time

# class for 

def tune_individually(X, W, y, n_iter=1e5):
  start = time.time()
  # Define hyperparameter distributions
  # X learner
  params_X_mu = {#'max_samples': [0.1*(i+1) for i in range(10)],
                 'max_samples': np.arange(0.1, 1.05, 0.05),
                 'max_features': range(1, len(X.columns)+1),
                 # 'min_samples_leaf': np.round(beta(1,4)*100) + 1}
                 'min_samples_leaf': range(1, 31)}
  params_X_tau = {#'max_samples': [0.1*(i+1) for i in range(10)],
                  'max_samples': np.arange(0.1, 1.05, 0.05),
                  'max_features': range(1, len(X.columns)+1),
                  # 'min_samples_leaf': np.round(beta(1,4)*100) + 1}
                  'min_samples_leaf': range(1, 31)}
  params_X_g = {#'max_samples': [0.1*(i+1) for i in range(10)],
                'max_samples': np.arange(0.1, 1.05, 0.05),
                'max_features': range(1, len(X.columns)+1),
                # 'min_samples_leaf': np.round(beta(1,4)*100) + 1}
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

  search_mu0_X = RandomizedSearchCV(mu0_base_X, params_X_mu, n_iter=n_iter, 
                                    scoring='neg_mean_squared_error', random_state=42)   
  search_mu1_X = RandomizedSearchCV(mu1_base_X, params_X_mu, n_iter=n_iter, 
                                    scoring='neg_mean_squared_error', random_state=42)
  search_tau0_X = RandomizedSearchCV(tau0_base_X, params_X_tau, n_iter=n_iter,
                                     scoring='neg_mean_squared_error', random_state=42)
  search_tau1_X = RandomizedSearchCV(tau1_base_X, params_X_tau, n_iter=n_iter,
                                     scoring='neg_mean_squared_error', random_state=42)
  search_g_X = RandomizedSearchCV(g_base_X, params_X_g, n_iter=n_iter, 
                                  scoring='neg_mean_squared_error', random_state=42)
  search_mu0_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, 
                                    scoring='neg_mean_squared_error', random_state=42)
  search_mu1_T = RandomizedSearchCV(mu0_base_T, params_T_mu, n_iter=n_iter, 
                                    scoring='neg_mean_squared_error', random_state=42)
  search_mu_S = RandomizedSearchCV(mu_base_S, params_S_mu, n_iter=n_iter, 
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

  best_params = {'mu0_base_X': search_mu0_X.best_params_, 
                 'mu1_base_X': search_mu1_X.best_params_,
                 'tau0_base_X': search_tau0_X.best_params_,
                 'tau1_base_X': search_tau1_X.best_params_,
                 'g_base_X': search_g_X.best_params_,
                 'mu0_base_T': search_mu0_T.best_params_,
                 'mu0_base_T': search_mu1_T.best_params_,
                 'mu_base_S': search_mu_S.best_params_}
  print('total execution time ', str(time.time() - start))
  return best_params

def main():
  # Set root directory
  base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

  # Read in data
  train_size = 20000
  sim = 'simA'
  i = 0
  samp_train_name = 'samp' + str(i+1) + '_train.parquet'
  # samp_test_name = 'samp' + str(i+1) + '_test.parquet'
  train = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_train_name)
  # test = pd.read_parquet(base_repo_dir / 'data/' / str(train_size) / sim / samp_test_name)

  # Data preprocessing
  X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
  y_train = train['Y']
  W_train = train['treatment']
  # X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
  # y_test = test['Y']
  # W_test = test['treatment']

  # Get best hyperparameters
  best_params = tune_individually(X_train, W_train, y_train, n_iter=1000)
  print(best_params)
  with open('best_params.json', 'w') as fp:
    json.dump(best_params, fp)

  return

if __name__ == "__main__":
  main()
