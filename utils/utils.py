#!/usr/bin/env python3


'''
Basic utility functions to support CATE estimation via metalearners

Alene Rhea, May 2021
'''

import pandas as pd
import numpy as np
from configClass import Tconfig, Sconfig, Xconfig

def strat_sample(df, n, replace=False, seed=42):
    '''
    df must contain binary column called "treatment" 
    Separately samples treatment (1) & controls (0) at same rate

    returns df of length n, with same treatment:control ratio as df
    '''

    # Isolate treatment and control groups from the training set
    df_treat = df[df.treatment==1]
    df_ctrl = df[df.treatment==0]

    # Set consistent fraction based on ratio of n to original # of items
    frac = n/len(df)

    # Sample treat and control separately at the same rate
    samp_treat = df_treat.sample(frac=frac, replace=replace, random_state=seed)
    samp_ctrl = df_ctrl.sample(frac=frac, replace=replace, random_state=seed)

    # Return sampled treatment and control items concatenated
    # (Ordering of items no longer random)
    return pd.concat([samp_treat, samp_ctrl])


def split_Xy_train_test(train, test, g_true=False):

    '''
    Split train and test dfs into X_train, y_train, W_train, and X_test
    Also split g_true if necessary
    '''

    X_train = train.drop(columns=['treatment', 'Y', 'tau', 'pscore'])
    y_train = train['Y']
    W_train = train['treatment']
    X_test = test.drop(columns=['treatment', 'Y', 'tau', 'pscore'])

    if g_true:
        g_true = test['pscore'].to_numpy()
        return (X_train, y_train, W_train, X_test, g_true)

    return (X_train, y_train, W_train, X_test)


def config_from_json(meta, sim, meta_base_dict, hyperparams):
    '''
    meta (str) : Name of metaleaner. Must be one of: "T", "S", or "X"

    sim (str) : Name of data simulation. Must be one of: "A", "B", "C", "D", "E", or "F"

    meta_base_dict (dict) : nested dict of the form saved in configurations/base_learners
                            eg {"T": {"mu_0": "rf", "mu_1": "rf"}, 
                                "S": {"mu": "rf"}, 
                                "X":{"mu_0": "rf", "mu_1": "rf", "tau_0": "rf", "tau_1": "rf"}}

    hyperparams (dict) : hyperparameters specific to given config
                         nested dict of the form saved in configurations/hyperparameters
                            eg {"mu_0": {"simA": {"n_estimators": 200, "min_samples_split": 15}, 
                                          "simB": {"n_estimators": 300, "min_samples_split": 10}, 
                                          "simC": {"n_estimators": 100, "min_samples_split": 10}, 
                                          "simD": {"n_estimators": 200, "min_samples_split": 20}, 
                                          "simE": {"n_estimators": 100, "min_samples_split": 15}, 
                                          "simF": {"n_estimators": 300, "min_samples_split": 20}}, 
                                 "mu_1": {"simA": {"n_estimators": 100, "min_samples_split": 20}, 
                                          "simB": {"n_estimators": 200, "min_samples_split": 25}, 
                                          "simC": {"n_estimators": 100, "min_samples_split": 10}, 
                                          "simD": {"n_estimators": 400, "min_samples_split": 20}, 
                                          "simE": {"n_estimators": 100, "min_samples_split": 5}, 
                                          "simF": {"n_estimators": 300, "min_samples_split": 10}}}    
    '''

    # dict to match metalearner to subclass
    meta_config_dict = {'T': Tconfig,
                        'S':  Sconfig,
                        'X': Xconfig}

    # intialize config object with correct item from meta_base_dict
    config = meta_config_dict[meta](**meta_base_dict[meta])

    # set hyperparameters on config object
    config.set_all_hyperparams(hp_dict=hyperparams, sim=sim)

    # return ready-to-use config object
    return config