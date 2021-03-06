#!/usr/bin/env python3

import argparse
from econml.grf import RegressionForest
import os, pathlib
import numpy as np
import pandas as pd
import re
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import utils.configClass
from utils.metaClass import t_learner, s_learner, x_learner
from utils.utils import strat_sample, split_Xy_train_test, config_from_json

'''
Small run usage example: python learners.py --samples 3 --training_sizes 5000
                        --base_learner_filename base_learners_iw_g_logreg.json --hp_substr default
Currently fitting RF-based T, S, and X-Learner, predicting CATE for each row of test set, & saving MSE results.
'''

def fit_predict_mse(train, test, config, export_preds):
    '''
    Used in learners loop
    '''

    #data preprocessing
    X_train, y_train, W_train, X_test = split_Xy_train_test(train, test)
    
    if (config.metalearner=='T') or (config.metalearner=='S'):

        # fit T or S learner
        fitted = fit_metalearner(X_train, y_train, W_train, X_test, config)

        if export_preds:
            # Predict test-set CATEs
            export_df = fitted.predict(X=X_test, export_preds=True)

            # save values to export_df
            export_df['tau_true'] = test.tau
            export_df['y_true'] = test.Y
            export_df['W'] = test.treatment
            export_df['pscore_true'] = test.pscore
            tau_preds = export_df.tau_preds
        else:
            # Predict test-set CATEs
            tau_preds = fitted.predict(X_test, export_preds=False)
            export_df = None

    elif config.metalearner=='X':
        # Predict test set CATEs using predicted propensities
        fitted, g_pred, export_df_train = fit_metalearner(X_train, y_train, W_train,
                                          X_test, config, export_preds)
        if export_preds:
            export_df_test = fitted.predict(X=X_test, g=g_pred, export_preds=True)
            export_df_test['tau_true'] = test.tau
            export_df_test['y_true'] = test.Y
            export_df_test['W'] = test.treatment
            export_df_test['pscore_true'] = test.pscore

            tau_preds = export_df_test.tau_preds
        else:
            tau_preds = fitted.predict(X=X_test, g=g_pred, export_preds=False)
            export_df_test = None
        
        # Bundle export dfs
        # Will be (None, None) if export_preds==False
        export_df = (export_df_train, export_df_test)

    # Calculate MSE on test set
    mse = np.mean((tau_preds - test.tau)**2)
    return (mse, export_df)

def fit_predict_ci(train, test, config):
    '''
    Used to return predicted taus for confidence intervals
    '''

    #data preprocessing
    X_train, y_train, W_train, X_test = split_Xy_train_test(train, test)

    if (config.metalearner=='T') or (config.metalearner=='S'):
        fitted = fit_metalearner(X_train, y_train, W_train, X_test, 
                                config, export_preds=False)
        tau_preds = fitted.predict(X_test, export_preds=False)

    elif config.metalearner=='X':
        # Predict test set CATEs using predicted propensities
        fitted, g_pred, _ = fit_metalearner(X_train, y_train, 
                                            W_train, X_test, config, 
                                            export_preds=False)
        tau_preds = fitted.predict(X=X_test, g=g_pred, export_preds=False)
    return tau_preds
    

def fit_metalearner(X_train, y_train, W_train, X_test, 
                    config, export_preds=False):

    '''
    Functionized and generalized method 
    to initialize base learners and train metalearner.

    Relies on configuration class.
    '''

    # dictionary to match algo code to partial init call
    # update LR item to match Tamar's IW branch
    base_learners = {'rf': partial(RegressionForest, 
                                   honest=True, inference=False, n_jobs=4, random_state=42),
                     'lr': LinearRegression,
                     'logreg': partial(LogisticRegression,
                                        n_jobs=4,
                                        random_state=42, 
                                        max_iter=500),
                     'rfc': partial(RandomForestClassifier,
                                    n_jobs=4, random_state=42)}
                     
    if config.metalearner == 'T':
        
        # Isolate necessary hyperparameters
        mu0_params = config.mu_0.hyperparams
        mu1_params = config.mu_1.hyperparams
        g_params = config.g.hyperparams
        
        # Intialize base learners
        mu0_base = base_learners[config.mu_0.algo](**mu0_params)
        mu1_base = base_learners[config.mu_1.algo](**mu1_params)
        g_base = base_learners[config.g.algo](**g_params)

        #fit g using training data and predict propensities
        g_fit = g_base.fit(X=X_train, y=W_train)
        g_pred = g_fit.predict_proba(X_train)[:, 1]

        # initialize and fit metalearner
        T = t_learner(mu0_base=mu0_base, mu1_base=mu1_base)
        T.fit(X=X_train, W=W_train, y=y_train, g=g_pred)
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
        g_pred_train = g_fit.predict_proba(X_train)[:, 1]
        g_pred_test = g_fit.predict_proba(X_test)[:, 1]
        
        # initialize metalearner
        X_learner = x_learner(mu0_base=mu0_base, mu1_base=mu1_base, 
                                tau0_base=tau0_base, tau1_base=tau1_base)

        # Fit treatment and response estimators mu0 and  mu1
        # export_df_train will be None-type if export_preds == False
        export_df_train = X_learner.fit(X=X_train, W=W_train, y=y_train, 
                                        g=g_pred_train, export_preds=export_preds)

        # return fitted X-learner and predicted propensities
        return(X_learner, g_pred_test, export_df_train)


def main(args):
    
    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]
    
    # read in tuned hyperparameter files
    rf_t = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_t_{}.json'.format(args.hp_substr)))
    rf_s = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_s_{}.json'.format(args.hp_substr)))
    rf_x = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_x_{}.json'.format(args.hp_substr)))
    
    rf_params = {'T': rf_t, 'S': rf_s, 'X': rf_x}
    
    # read in file with base learner model types for each metalearner
    meta_base_dict = json.load(open(base_repo_dir / 'configurations' / 'base_learners' / args.base_learner_filename))
    
    # Base-learners filename substring (for descriptive output filenames)
    bl_substr = re.search('base_learners_(.*?).json', args.base_learner_filename).group(1)

    #initialize temp list where results will be stored and column names for results df
    rows=[]
    for sim in ['simA', 'simB', 'simC', 'simD', 'simE', 'simF']:
        print('---------------------------')
        print('Starting '+ sim)
        for i in range(args.samples):
            print('     Starting sample'+ str(i+1))
            samp_train_name = 'samp' + str(i+1) + '_train.parquet'
            samp_test_name = 'samp' + str(i+1) + '_test.parquet'
            full_train = pd.read_parquet(base_repo_dir / 'data' / sim / samp_train_name)
            test = pd.read_parquet(base_repo_dir / 'data' / sim / samp_test_name)
            for train_size in args.training_sizes:
                print('         Training set size:', train_size)
                if train_size != 300000:
                    train = strat_sample(full_train, n=train_size, replace=False, seed=42)
                else: 
                    train = full_train
                
                # initialize dictionary to hold the 3 MSEs for this training size
                mses=dict()

                for metalearner in meta_base_dict.keys():
                    
                    # Create configuration object
                    config = config_from_json(meta=metalearner, 
                                                sim=sim[-1], meta_base_dict=meta_base_dict, 
                                                hyperparams=rf_params[metalearner])
                    
                    # Export predictions for first sample of largest size
                    # if export_preds flag is set to True
                    if (args.export_preds) and (i == 0) and (train_size == 300000):
                        export_dir = os.path.join(base_repo_dir, 'results', 'preds')
                        if not os.path.exists(export_dir):
                            os.makedirs(export_dir)
                        mse, export_df = fit_predict_mse(train, test, 
                                                         config, 
                                                        export_preds=True)
                        if metalearner=='X':
                            print("Exporting X learner train and test preds dataframes.")
                            # parse export_df tuple
                            export_df_train = export_df[0]
                            export_df_test = export_df[1]
                            
                            # Build descriptive filenames
                            filename_train = 'X_'+ sim + '_' + bl_substr + \
                                             '_' + args.hp_substr + '_train_preds.parquet'

                            filename_test = 'X_'+ sim + '_' + bl_substr + \
                                             '_' + args.hp_substr + '_test_preds.parquet'
                            
                            # Save train preds and test preds
                            export_df_train.to_parquet(os.path.join(export_dir, filename_train))
                            export_df_test.to_parquet(os.path.join(export_dir, filename_test))
                        else:
                            print("Exporting ", metalearner, " learner test preds dataframes.")
                            filename_df = metalearner + '_' + sim + '_' + bl_substr + \
                                             '_' + args.hp_substr + '_test_preds.parquet'
                            # Save test preds
                            export_df.to_parquet(os.path.join(export_dir, filename_df))
                    else:
                        mse, _ = fit_predict_mse(train, test, 
                                                         config, export_preds=False)

                    # Save MSE to dict
                    mses[metalearner] = mse

                rows.append([sim, i, train_size, mses['T'], mses['S'], mses['X']])

        print('End of ', sim)
        # Export intermediary results
        columns=['simulation', 'trial', 'n', 'T_mse', 'S_mse', 'X_mse']
        results_full_sim = pd.DataFrame(rows, columns=columns)
        results_sim = results_full_sim.groupby(['simulation', 'n'])[
                        'T_mse', 'S_mse', 'X_mse'].mean().reset_index()

        # Save results with filename of the form results_{A}_{B}_{C}_{D}.csv where A is the substring from base_learners
        # filename, B is the substring from hyperpameter filenames, C is # of samples, and D is sim.
        filename_str = bl_substr + '_' + args.hp_substr + '_' + str(args.samples)
        filename_full_sim = 'results_full_' + filename_str +  '_' + sim + '.csv'
        filename_results_sim = 'results_' + filename_str + '_' + sim + '.csv'
        mse_out_dir = os.path.join(base_repo_dir, 'results', 'mse')
        if not os.path.exists(mse_out_dir):
            os.makedirs(mse_out_dir)
        results_sim.to_csv(os.path.join(mse_out_dir, filename_results_sim), index=False)
        results_full_sim.to_csv(os.path.join(mse_out_dir, filename_full_sim), index=False)


    # Export all results at the end of all sims
    results_full = pd.DataFrame(rows, columns=columns)
    results = results_full.groupby(['simulation', 'n'])['T_mse', 'S_mse', 'X_mse'].mean().reset_index()
    print('---------------------------')
    print('Results:\n', results)

    # Save results with filename of the form results_{A}_{B}_{C}.csv where A is the substring from base_learners
    # filename, B is the substring from hyperpameter filenames, and C is # of samples.
    filename_full = 'results_full_' + filename_str + '.csv'
    filename_results = 'results_' + filename_str + '.csv'
    results.to_csv(os.path.join(mse_out_dir, filename_results), index=False)
    results_full.to_csv(os.path.join(mse_out_dir, filename_full), index=False)

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
    parser.add_argument("--base_learner_filename", type=str, default='base_learners.json',
                        help='Name of base learner file to use. Should be of form base_learners_XX.json ' +
                        'and reside in configurations/base_learners')
    parser.add_argument("--hp_substr", type=str, default='default',
                        help='The naming convention for the hyperparameter files that should be used. For ' +
                        'example if user wishes to use rf_t_default.json/rf_s_default.json/etc. then the string ' +
                        'passed should be default.')
    args = parser.parse_args()

    # Call main routine
    main(args)
