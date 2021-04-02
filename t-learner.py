#!/usr/bin/env python3

from bartpy.sklearnmodel import SklearnModel
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import numpy as np
import pandas as pd

'''
From Rosenberg Slides:
    µ^0(x) = E[Y (0) | X = x]
    µ^1(x) = E[Y (1) | X = x]
    τˆT (x) = µˆ1(x) − µˆ0(x)


Currently fitting RF-based T-Learner on 1st sample of sim A,
and predicting CATE for each row of test set.
Does not save predictions or fit. Only prints RMSE.

This script needs to be plugged into a broader pipeline,
and adapted accordingly.

authors: Alene Rhea and Tamar Novetsky, April 1 2021
'''


class t_learner:

    '''
    Current implementation depends on base learners 
    having .fit and .predict methods

    bartpy offers these methods, but is slow.
    May need to use different BART implementation,
    and then update our class functions accordingly.
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

# intialize base learner
# bart = SklearnModel() # takes too long
rf = RegressionForest(honest=True, random_state=42)

# initialize metalearner
# T = t_learner(mu0_base=bart, mu1_base=bart) # takes too long
T = t_learner(mu0_base=rf, mu1_base=rf)

# Set root directory
base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

# Read in data
train = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_train.csv')
test = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_test.csv')
X_train = train.drop(columns=['treatment', 'Y', 'tau'])
y_train = train['Y']
W_train = train['treatment']
X_test = test.drop(columns=['treatment', 'Y', 'tau'])

# Fit T-learner
T.fit(X=X_train, W=W_train, y=y_train)

# Predict test-set CATEs
tau_preds = T.predict(X=X_test)

# Calculate RMSE on test set
rmse = np.sqrt(np.mean((tau_preds - test.tau)**2))

print('tau_preds.shape: ', len(tau_preds)) # should be (1000,)
print('RMSE: ', rmse)