#!/usr/bin/env python3


'''
Basic utility functions to support CATE estimation via metalearners

Alene Rhea, May 2021
'''

import pandas as pd
import numpy as np

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