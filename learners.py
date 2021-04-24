#!/usr/bin/env python3

import argparse
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression

'''
Small run usage example: python learners.py --samples 3 --training_sizes 5000 10000
Currently fitting RF-based T, S, and X-Learner and predicting CATE for each row of test set.
Does not save predictions or fit. Only prints and saves MSE.
'''


class t_learner:

    '''
    From Rosenberg Slides:
        µ^0(x) = E[Y (0) | X = x]
        µ^1(x) = E[Y (1) | X = x]
        τˆT (x) = µˆ1(x) − µˆ0(x)

    authors: Alene Rhea and Tamar Novetsky, April 1 2021
    '''

    def __init__(self, mu0_base, mu1_base):
        # Make copies of initialized base learners
        self.mu0_base = clone(mu0_base, safe=False)
        self.mu1_base = clone(mu1_base, safe=False)

    def fit(self, X, W, y):
        # Call fit methods on each base learner
        self.mu0_base.fit(X[W==0], y[W==0])
        self.mu1_base.fit(X[W==1], y[W==1])

    def predict(self, X):
        y1_preds = self.mu1_base.predict(X)
        y0_preds = self.mu0_base.predict(X)
        tau_preds = y1_preds - y0_preds
        return tau_preds.flatten()


class s_learner:

    '''
    From Rosenberg Slides:
        µ^(x,w) = E[Y | X = x, W = w]
        τˆS (x) = µˆ(x,1) − µˆ(x,0)

    authors: Kelsey Markey and Lauren D'Arinzo, April 4 2021
    '''

    def __init__(self, mu_base):
        # Make copies of initialized base learner
        self.mu_base = clone(mu_base, safe=False)

    def fit(self, X_W, y):
        # Call fit method
        self.mu_base.fit(X_W, y)

    def predict(self, X):
        X_W_1 = X.copy(deep=True)
        X_W_1["W"] = pd.Series(np.ones(len(X_W_1)))
        y1_preds = self.mu_base.predict(X_W_1)

        X_W_0 = X.copy(deep=True)
        X_W_0["W"] = pd.Series(np.zeros(len(X_W_0)))
        y0_preds = self.mu_base.predict(X_W_0)

        tau_preds = y1_preds - y0_preds
        return tau_preds.flatten()


class x_learner:

    '''
    From Rosenberg Slides:
        µ^0(x) = E[Y (0) | X = x]
        µ^1(x) = E[Y (1) | X = x]
        τˆx (x) = g(x)t_0ˆ(x) − (1-g(x))t_1ˆ(x)

    authors: Kelsey Markey and Lauren D'Arinzo, April 4 2021
    '''

    def __init__(self, mu0_base, mu1_base, tau0_base, tau1_base):
        # Make copies of initialized base learner
        self.mu0_base = clone(mu0_base, safe=False)
        self.mu1_base = clone(mu1_base, safe=False)
        self.tau0_base = clone(tau0_base, safe=False)
        self.tau1_base = clone(tau1_base, safe=False)

    def fit(self, X, W, y):
        # Call fit method
        self.mu0_base.fit(X[W==0], y[W==0])
        self.mu1_base.fit(X[W==1], y[W==1])
    
        #Impute y0 for treated group using mu0
        y0_treat=self.mu0_base.predict(X[W==1]).flatten()
        imputed_TE_treatment = y[W==1] - y0_treat

        #Impute y1 for control group using mu1
        y1_control=self.mu1_base.predict(X[W==0]).flatten()
        imputed_TE_control = y1_control - y[W==0] 

        #Fit tau0: CATE estimate fit to the control group
        self.tau0_base.fit(X[W==0], imputed_TE_control)

        #Fit tau1: CATE estimate fit to the treatment group
        self.tau1_base.fit(X[W==1], imputed_TE_treatment)

    def predict(self, X, g):
        '''
        g : weight vector that should be length of the test set
        '''
        tau0_preds = self.tau0_base.predict(X).flatten()
        tau1_preds = self.tau1_base.predict(X).flatten()
        tau_preds = (g.T * tau0_preds) + ((1-g).T*tau1_preds)
        return tau_preds.flatten()


def fit_get_mse_t(train, test, mu0_base, mu1_base):
    
    '''
    mu0_base: base learner that has already been initialized
    mu1_base: base learner that has already been initialized
    '''

    #data preprocessing
    X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    y_train = train['Y']
    W_train = train['treatment']
    X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])

    #initialize metalearner
    T = t_learner(mu0_base=mu0_base, mu1_base=mu1_base)
    T.fit(X=X_train, W=W_train, y=y_train)
    
    # Predict test-set CATEs
    tau_preds = T.predict(X=X_test)

    # Calculate MSE on test set
    mse = np.mean((tau_preds - test.tau)**2)
    return mse


def fit_get_mse_s(train, test, mu_base):
    
    '''
    mu_base: base learner that has already been initialized
    '''

    #data preprocessing
    X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    y_train = train['Y']
    W_train = train['treatment']
    X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])

    #initialize metalearner
    S = s_learner(mu_base=mu_base)
    
    #fit S-learner
    X_W = pd.concat([X_train, W_train], axis=1)
    S.fit(X_W=X_W, y=y_train)
    
    # Predict test-set CATEs
    tau_preds = S.predict(X=X_test)

    # Calculate MSE on test set
    mse = np.mean((tau_preds - test.tau)**2)
    return mse


def fit_get_mse_x(train, test, mu0_base, mu1_base, tau0_base, tau1_base):
    
    '''
    mu0_base: base learner that has already been initialized
    mu1_base: base learner that has already been initialized
    tau0_base: base learner that has already been initialized
    tau1_base: base learner that has already been initialized
    '''

    #data preprocessing
    X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    y_train = train['Y']
    W_train = train['treatment']
    X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    g_true = test['pscore'].to_numpy()
    
    #fit g using training data
    g_fit = LogisticRegression(fit_intercept=True, max_iter=2000).fit(
       X=X_train, y=W_train)
    #predict on test set
    g_pred = g_fit.predict_proba(X_test)[:, 1]
    
    # initialize metalearner
    X_learner = x_learner(mu0_base=mu0_base, mu1_base=mu1_base, tau0_base=tau0_base, tau1_base=tau1_base)
    # Fit treatment and response estimators mu0 and  mu1
    X_learner.fit(X=X_train, W=W_train, y=y_train)
    
    # Predict test set CATEs using true and predicted propensities
    tau_preds_gtrue = X_learner.predict(X=X_test, g=g_true)
    tau_preds_gpred = X_learner.predict(X=X_test, g=g_pred)
    
    # Calculate MSE on test set for X-Learners with true and predicted propensities
    mse_true = np.mean((tau_preds_gtrue - test.tau)**2)
    mse_pred = np.mean((tau_preds_gpred - test.tau)**2)
    
    return mse_true, mse_pred


def main(samples=30, training_sizes=[5000, 10000, 20000, 100000, 300000]):
    
    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]
    
    # read in tuned hyperparameter files
    # TODO: add in other base learner
    rf_t = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_t.json'))
    rf_s = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_s.json'))
    rf_x = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_x.json'))
    
    rf_params = {'T': rf_t, 'S': rf_s, 'X': rf_x}
    
    # read in base learner model types for each metalearner
    meta_base_dict = json.load(open(base_repo_dir / 'configurations' / 'base_learners' / 'base_learners.json'))
    
    #initialize temp list where results will be stored and column names for results df
    rows=[]

    for train_size in training_sizes:
        print('---------------------------')
        print('Training set size:', train_size)
        
        for sim in ['simA', 'simB', 'simC', 'simD', 'simE', 'simF']:
            #print('---------------------------')
            print('     Starting '+ sim)

            T_RF = []
            S_RF = []
            X_RF_PTrue = []
            X_RF_PPred = []

            for i in range(samples):
                #print('')
                samp_train_name = 'samp' + str(i+1) + '_train.parquet'
                samp_test_name = 'samp' + str(i+1) + '_test.parquet'
                train = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_train_name)
                test = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_test_name)
                
                for metalearner in meta_base_dict.keys():
                    
                    if metalearner == 'T':
                        for base_learner_dict in meta_base_dict[metalearner]:
                            #print(sim+'/' +'sample_'+str(i+1)+'/'+metalearner+'-learner:')
                            if base_learner_dict['mu_0'] == 'rf':
                                mu0_hyperparams = rf_params[metalearner]['mu_0'][sim]
                                # uncomment out all of these lines once hyperparameter jsons updated from tuning
                                # mu0_base = RegressionForest(honest=True, random_state=42, **mu0_hyperparams)
                                mu0_base = RegressionForest(honest=True, random_state=42)
                            if base_learner_dict['mu_1'] == 'rf':
                                mu1_hyperparams = rf_params[metalearner]['mu_1'][sim]
                                #mu1_base = RegressionForest(honest=True, random_state=42, **mu1_hyperparams)
                                mu1_base = RegressionForest(honest=True, random_state=42)

                            # TODO: add logic for if base_learner_dict[mu_0]/[mu_1] is other base learner type
                            mse = fit_get_mse_t(train, test, mu0_base, mu1_base)
                            T_RF.append(mse)
                            #print('     MSE=', mse)

                    if metalearner == 'S':
                        for base_learner_dict in meta_base_dict[metalearner]:
                            #print(sim+'/' +'sample_'+str(i+1)+'/'+ metalearner+'-learner:')
                            if base_learner_dict['mu'] == 'rf':
                                mu_hyperparams = rf_params[metalearner]['mu'][sim]
                                # uncomment out all of these lines once hyperparameter jsons updated from tuning
                                # mu_base = RegressionForest(honest=True, random_state=42, **mu_hyperparams)
                                mu_base = RegressionForest(honest=True, random_state=42)
                            
                            # TODO: add logic for if base_learner_dict[mu] is other base learner type
                            mse = fit_get_mse_s(train, test, mu_base)
                            S_RF.append(mse)
                            #print('     MSE=', mse)
                            
                    if metalearner == 'X':
                        for base_learner_dict in meta_base_dict[metalearner]:
                            #print(sim+'/' +'sample_'+str(i+1)+'/'+ metalearner+'-learner:')
                            if base_learner_dict['mu_0'] == 'rf':
                                mu0_hyperparams = rf_params[metalearner]['mu_0'][sim]
                                # uncomment out all of these lines once hyperparameter jsons updated from tuning
                                # mu0_base = RegressionForest(honest=True, random_state=42, **mu0_hyperparams)
                                mu0_base = RegressionForest(honest=True, random_state=42)
                            if base_learner_dict['mu_1'] == 'rf':
                                mu1_hyperparams = rf_params[metalearner]['mu_1'][sim]
                                # mu1_base = RegressionForest(honest=True, random_state=42, **mu1_hyperparams)
                                mu1_base = RegressionForest(honest=True, random_state=42)
                            if base_learner_dict['tau_0'] == 'rf':
                                tau0_hyperparams = rf_params[metalearner]['tau_0'][sim]
                                # tau0_base = RegressionForest(honest=True, random_state=42, **tau0_hyperparams)
                                tau0_base = RegressionForest(honest=True, random_state=42)
                            if base_learner_dict['tau_1'] == 'rf':
                                tau1_hyperparams = rf_params[metalearner]['tau_1'][sim]
                                # tau1_base = RegressionForest(honest=True, random_state=42, **tau1_hyperparams)
                                tau1_base = RegressionForest(honest=True, random_state=42)
                            
                            # TODO: add logic for if other base learner types
                            mse_true, mse_pred = fit_get_mse_x(train, test, mu0_base, mu1_base, tau0_base, tau1_base)
                            X_RF_PTrue.append(mse_true)
                            X_RF_PPred.append(mse_pred)
                            #print('     MSE (true pscore)=', mse_true)
                            #print('     MSE (estimated pscore)=', mse_pred)   
            rows.append([sim, train_size, np.mean(T_RF), np.mean(S_RF), np.mean(X_RF_PTrue), np.mean(X_RF_PPred)])
            #print(rows)                             
    columns=['simulation', 'n', 'T_RF', 'S_RF', 'X_RF_PTrue', 'X_RF_PPred']
    results = pd.DataFrame(rows, columns=columns)
    results.sort_values(by=['simulation', 'n'], inplace=True)
    print('---------------------------')
    print('Results:\n', results)
    results.to_csv('results.csv')
    return

if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30,
                        help='Number of samples to read in from data directory')
    parser.add_argument("--training_sizes", nargs='+', type=int, default=[5000, 10000, 20000, 100000, 300000],
                        help='Training set sizes to read in from data directory')
    args = parser.parse_args()

    # Call main routine
    main(samples=args.samples, training_sizes=args.training_sizes)
