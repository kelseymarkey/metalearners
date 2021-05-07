#!/usr/bin/env python3

import argparse
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import numpy as np
import pandas as pd
import re
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from configClass import config, Tconfig, Sconfig, Xconfig, baseLearner

'''
Small run usage example: python learners.py --samples 3 --training_sizes 5000 --base_learner_filename base_learners_rf.json
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

    def predict(self, X, export_preds=False):
        y1_preds = self.mu1_base.predict(X)
        y0_preds = self.mu0_base.predict(X)
        tau_preds = y1_preds - y0_preds

        if export_preds:
            export_df = X.copy(deep=True)
            export_df["y1_preds"] = y1_preds.flatten()
            export_df["y0_preds"] = y0_preds.flatten()
            export_df["tau_preds"] = tau_preds.flatten()
            return export_df
        else:
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

    def predict(self, X, export_preds=False):
        X_W_1 = X.copy(deep=True)
        X_W_1["W"] = pd.Series(np.ones(len(X_W_1)))
        y1_preds = self.mu_base.predict(X_W_1)

        X_W_0 = X.copy(deep=True)
        X_W_0["W"] = pd.Series(np.zeros(len(X_W_0)))
        y0_preds = self.mu_base.predict(X_W_0)

        tau_preds = y1_preds - y0_preds

        if export_preds:
            export_df = X.copy(deep=True)
            export_df["y1_preds"] = y1_preds.flatten()
            export_df["y0_preds"] = y0_preds.flatten()
            export_df["tau_preds"] = tau_preds.flatten()
            return export_df
        else:
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

    def fit(self, X, W, y, export_preds):
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

        if export_preds:
            # Export y0_treat and y1_control predictions on training set
            y0_treat_df = X[W==1].copy(deep=True)
            y0_treat_df["y0_treat"] = y0_treat
            y0_treat_df["y"] = y[W==1]
            y0_treat_df["W"] = 1

            y1_control_df = X[W==0].copy(deep=True)
            y1_control_df["y1_control"] = y1_control
            y1_control_df["y"] = y[W==0]
            y1_control_df["W"] = 0

            export_df_train = y0_treat_df.append(y1_control_df, ignore_index=True, sort=False)
        else:
            export_df_train = None
        return export_df_train


    def predict(self, X, g):
        '''
        g : weight vector that should be length of the test set
        '''
        tau0_preds = self.tau0_base.predict(X).flatten()
        tau1_preds = self.tau1_base.predict(X).flatten()
        tau_preds = (g.T * tau0_preds) + ((1-g).T*tau1_preds)
        return tau_preds.flatten()


def fit_get_mse_t(train, test, mu0_base, mu1_base, export_preds=False):
    
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
    
    if export_preds:
        # Predict test-set CATEs
        export_df = T.predict(X=X_test, export_preds=export_preds)
        tau_preds = export_df.tau_preds
    else:
        # Predict test-set CATEs
        tau_preds = T.predict(X=X_test, export_preds=export_preds)
        export_df = None
    # Calculate MSE on test set
    mse = np.mean((tau_preds - test.tau)**2)
    return (mse, export_df)


def fit_get_mse_s(train, test, mu_base, export_preds=False):
    
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

    if export_preds:
        # Predict test-set CATEs
        export_df = S.predict(X=X_test, export_preds=export_preds)
        tau_preds = export_df.tau_preds
    else:
        # Predict test-set CATEs
        tau_preds = S.predict(X=X_test, export_preds=export_preds)
        export_df = None
    # Calculate MSE on test set
    mse = np.mean((tau_preds - test.tau)**2)
    return (mse, export_df)


def fit_get_mse_x(train, test, mu0_base, mu1_base, tau0_base, tau1_base, rf_prop=False, export_preds=False):
    
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
    if rf_prop:
        g_fit = RandomForestClassifier(random_state=0).fit(X=X_train, y=W_train)
    else:
        g_fit = LogisticRegression(fit_intercept=True, max_iter=2000, random_state=42).fit(
            X=X_train, y=W_train)
    #predict on test set
    g_pred = g_fit.predict_proba(X_test)[:, 1]
    
    # initialize metalearner
    X_learner = x_learner(mu0_base=mu0_base, mu1_base=mu1_base, tau0_base=tau0_base, tau1_base=tau1_base)
    # Fit treatment and response estimators mu0 and  mu1
    export_df_train = X_learner.fit(X=X_train, W=W_train, y=y_train, export_preds=export_preds)

    # Predict test set CATEs using true and predicted propensities
    tau_preds_gtrue = X_learner.predict(X=X_test, g=g_true)
    tau_preds_gpred = X_learner.predict(X=X_test, g=g_pred)

    if export_preds:
        export_df = X_test.copy(deep=True)
        export_df["g_pred"] = g_pred
        export_df["tau_preds_gtrue"] = tau_preds_gtrue
        export_df["tau_preds"] = tau_preds_gpred
    else:
        export_df = None
    
    # Calculate MSE on test set for X-Learners with true and predicted propensities
    mse_true = np.mean((tau_preds_gtrue - test.tau)**2)
    mse_pred = np.mean((tau_preds_gpred - test.tau)**2)
    
    return (mse_true, mse_pred, export_df, export_df_train)

def split_Xy_train_test(train, test, g_true=False):

    X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    y_train = train['Y']
    W_train = train['treatment']
    X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])

    if g_true:
        g_true = test['pscore'].to_numpy()
        return (X_train, y_train, W_train, X_test, g_true)

    return (X_train, y_train, W_train, X_test)


def fit_predict(train, test, config):

    #data preprocessing
    X_train, y_train, W_train, X_test = split_Xy_train_test(train, test)

    if (config.metalearner=='T') or (config.metalearner=='S'):
        fitted = fit_metalearner(X_train, y_train, W_train, X_test, config)
        tau_preds = fitted.predict(X_test, export_preds=False)

    elif config.metalearner=='X':
        # Predict test set CATEs using predicted propensities
        fitted, g_pred = fit_metalearner(X_train, y_train, W_train,
                                         X_test, config)
        tau_preds = fitted.predict(X=X_test, g=g_pred)
    return tau_preds
    

def fit_metalearner(X_train, y_train, W_train, X_test, config):

    '''
    To be called by conf_int.py

    Functionized and generalized method 
    to initialize base learners and train metalearner.

    Return predictions on test set
    No need to calculate MSE

    Could be wrapper around general intialize / train func(s);
    with test set prediction added on at the end    
    '''

    # dictionary to match algo code to partial init call
    # update LR item to match Tamar's IW branch
    base_learners = {'rf': partial(RegressionForest, 
                                   honest=True, random_state=42),
                     'lr': LinearRegression,
                     'logreg': partial(LogisticRegression, 
                                        random_state=42),
                     'rfc': partial(RandomForestClassifier, 
                                   random_state=42)}
                     
    if config.metalearner == 'T':
        
        # Isolate necessary hyperparameters
        mu0_params = config.mu_0.hyperparams
        mu1_params = config.mu_1.hyperparams
        
        # Intialize base learners
        mu0_base = base_learners[config.mu_0.algo](**mu0_params)
        mu1_base = base_learners[config.mu_1.algo](**mu1_params)

        # initialize and fit metalearner
        T = t_learner(mu0_base=mu0_base, mu1_base=mu1_base)
        T.fit(X=X_train, W=W_train, y=y_train)
        return T
 
    if config.metalearner == 'S':

        # Isolate necessary hyperparameters
        mu_params = config.mu.hyperparams
        
        # Intialize base learners
        mu_base = base_learners[config.mu.algo](**mu_params)

        #initialize and fit metalearner
        S = s_learner(mu_base=mu_base)
        X_W = pd.concat([X_train, W_train], axis=1)
        S.fit(X_W=X_W, y=y_train)
        return S
        
    if config.metalearner == 'X':

        # Isolate necessary hyperparameters
        mu0_params = config.mu_0.hyperparams
        mu1_params = config.mu_1.hyperparams
        tau0_params = config.tau_0.hyperparams
        tau1_params = config.tau_1.hyperparams
        g_params = config.g.hyperparams

        # Intialize base learners
        mu0_base = base_learners[config.mu_0.algo](**mu0_params)
        mu1_base = base_learners[config.mu_1.algo](**mu1_params)
        tau0_base = base_learners[config.tau_0.algo](**tau0_params)
        tau1_base = base_learners[config.tau_0.algo](**tau1_params)
        g_base = base_learners[config.g.algo](**g_params)

        #fit g using training data and predict propensities
        g_fit = g_base.fit(X=X_train, y=W_train)
        g_pred = g_fit.predict_proba(X_test)[:, 1]
        
        # initialize metalearner
        X_learner = x_learner(mu0_base=mu0_base, mu1_base=mu1_base, 
                                tau0_base=tau0_base, tau1_base=tau1_base)

        # Fit treatment and response estimators mu0 and  mu1
        X_learner.fit(X=X_train, W=W_train, y=y_train, export_preds=False)

        # return fitted X-learner and predicted propensities
        return(X_learner, g_pred)


def main(args):
    
    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]
    
    # read in tuned hyperparameter files
    # TODO: add in other base learner
    rf_t = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_t.json'))
    rf_s = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_s.json'))
    rf_x = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_x.json'))
    
    rf_params = {'T': rf_t, 'S': rf_s, 'X': rf_x}
    
    # read in file with base learner model types for each metalearner
    meta_base_dict = json.load(open(base_repo_dir / 'configurations' / 'base_learners' / args.base_learner_filename))
    
    #initialize temp list where results will be stored and column names for results df
    rows=[]

    for train_size in args.training_sizes:
        print('---------------------------')
        print('Training set size:', train_size)
        
        for sim in ['simA', 'simB', 'simC', 'simD', 'simE', 'simF']:
            print('     Starting '+ sim)

            # Instantiate empty lists for saving mse results
            T_results, S_results, X_results_PTrue, X_results_PPred= [], [], [], []

            for i in range(args.samples):
                samp_train_name = 'samp' + str(i+1) + '_train.parquet'
                samp_test_name = 'samp' + str(i+1) + '_test.parquet'
                train = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_train_name)
                test = pd.read_parquet(base_repo_dir / 'data' / str(train_size) / sim / samp_test_name)

                for metalearner in meta_base_dict.keys():
                    base_learner_dict = meta_base_dict['metalearner'][0]
                    if metalearner == 'T':
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

                        if (args.export_preds and i == 0):
                            (mse, export_df) = fit_get_mse_t(train, test, mu0_base, mu1_base, export_preds=True)
                        else:
                            (mse, _) = fit_get_mse_t(train, test, mu0_base, mu1_base, export_preds=False)
                        T_results.append(mse)


                    if metalearner == 'S':
                        if base_learner_dict['mu'] == 'rf':
                            mu_hyperparams = rf_params[metalearner]['mu'][sim]
                            # uncomment out all of these lines once hyperparameter jsons updated from tuning
                            # mu_base = RegressionForest(honest=True, random_state=42, **mu_hyperparams)
                            mu_base = RegressionForest(honest=True, random_state=42)
                        
                        # TODO: add logic for if base_learner_dict[mu] is other base learner type
                        if (args.export_preds and i == 0):
                            (mse, export_df) = fit_get_mse_s(train, test, mu_base, export_preds=True)
                        else:
                            (mse, _) = fit_get_mse_s(train, test, mu_base, export_preds=False)
                        S_results.append(mse)

                        
                    if metalearner == 'X':
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

                        # Why only for X-learner?
                        if (args.export_preds and i == 0):
                            export_preds = True
                        else:
                            export_preds = False

                        (mse_true, mse_pred, export_df, export_df_train) = fit_get_mse_x(train, test,
                            mu0_base, mu1_base, tau0_base, tau1_base, args.rf_prop, export_preds)
                        X_results_PTrue.append(mse_true)
                        X_results_PPred.append(mse_pred)

                    if args.export_preds and i == 0 and train_size == 300000:
                        # Export predictions for first sample if export_preds flag. Only for largest sample size
                        export_dir = os.path.join(base_repo_dir, 'data', 'preds')
                        if not os.path.exists(export_dir):
                            os.makedirs(export_dir)
                        if metalearner == 'X':
                            # X learner has an extra file to export with Y0/Y1 preds on train
                            filename_train_preds = sim + '_X_' + str(train_size) + '_train_preds.parquet'
                            export_df_train.to_parquet(os.path.join(export_dir, filename_train_preds))
                        filename = sim + '_' + metalearner + '_' + str(train_size) + '.parquet'
                        export_df.to_parquet(os.path.join(export_dir, filename))


            rows.append([sim, train_size, np.mean(T_results), np.mean(S_results),
                np.mean(X_results_PTrue), np.mean(X_results_PPred)])

    columns=['simulation', 'n', 'T_mse', 'S_mse', 'X_mse_PTrue', 'X_mse_PPred']
    results = pd.DataFrame(rows, columns=columns)
    results.sort_values(by=['simulation', 'n'], inplace=True)
    print('---------------------------')
    print('Results:\n', results)

    # Save results
    substring = re.search('base_learners_(.*?).json', args.base_learner_filename).group(1)
    filename = 'results_' + substring + '.csv'
    results.to_csv(filename, index=False)

    return


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30,
                        help='Number of samples to read in from data directory')
    parser.add_argument("--training_sizes", nargs='+', type=int, default=[5000, 10000, 20000, 100000, 300000],
                        help='Training set sizes to read in from data directory')
    parser.add_argument("--export_preds", action='store_true',
                        help='Boolean flag indicating that predictions (e.g. y0_preds, y1_preds for T learner) ' +
                        'should be exported.')
    parser.add_argument("--rf_prop", action='store_true',
                        help='Boolean flag indicating that RandomForestClassifier should be used for ' +
                        'predicted propensity scores. Without flag LogisticRegression is used.')
    parser.add_argument("--base_learner_filename", type=str, default='base_learners.json',
                        help='Name of base learner file to use. Should be of form base_learners_XX.json ' +
                        'and reside in configurations/base_learners')
    args = parser.parse_args()

    # Call main routine
    main(args)
