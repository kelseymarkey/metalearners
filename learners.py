#!/usr/bin/env python3

from bartpy.sklearnmodel import SklearnModel
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import numpy as np
import pandas as pd

'''
Currently fitting RF-based T and S-Learner on 1st sample of sim A,
and predicting CATE for each row of test set.
Does not save predictions or fit. Only prints RMSE.

These scripts need to be plugged into a broader pipeline,
and adapted accordingly.

Current implementation depends on base learners 
having .fit and .predict methods

bartpy offers these methods, but is slow.
May need to use different BART implementation,
and then update our class functions accordingly.
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

# intialize base learner
# bart = SklearnModel() # takes too long
rf = RegressionForest(honest=True, random_state=42)

# initialize metalearner
S = s_learner(mu_base=rf)

# Set root directory
base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

# Read in data
train = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_train.csv')
test = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_test.csv')
X_train = train.drop(columns=['treatment', 'Y', 'tau'])
y_train = train['Y']
W_train = train['treatment']
X_test = test.drop(columns=['treatment', 'Y', 'tau'])

# Fit S-learner
X_W = pd.concat([X_train, W_train], axis=1)
S.fit(X_W=X_W, y=y_train)

# Predict test-set CATEs
tau_preds = S.predict(X=X_test)

# Calculate RMSE on test set
rmse = np.sqrt(np.mean((tau_preds - test.tau)**2))

print('tau_preds.shape: ', len(tau_preds)) # should be (1000,)
print('RMSE: ', rmse)

class x_learner:

    '''
    From Rosenberg Slides:
        µ^0(x) = E[Y (0) | X = x]
        µ^1(x) = E[Y (1) | X = x]


        τˆx (x) = g(x)t_0ˆ(x) − (1-g(x))t_1ˆ(x)

    authors: Kelsey Markey and Lauren D'Arinzo, April 4 2021
    '''

    def __init__(self, mu0_base, mu1_base, tau0_base, tau1_base, g):
        # Make copies of initialized base learner
        self.mu0_base = clone(mu0_base, safe=False)
        self.mu1_base = clone(mu1_base, safe=False)
        self.tau0_base = clone(tau0_base, safe=False)
        self.tau1_base = clone(tau1_base, safe=False)
        self.g = g

    def fit_t(self, X, W, y):
        # Call fit method
        self.mu0_base.fit(X[W==0], y[W==0])
        self.mu1_base.fit(X[W==1], y[W==1])
    
    def predict_impute(self, X, W, y):
        #impute y0 for treated group using mu0
        y0_treat=self.mu0_base.predict(X[W==1]).flatten()
        #impute y1 for control group using mu1
        y1_control=self.mu1_base.predict(X[W==0]).flatten()

        #### y1_control is all the same value, needs to be debugged! #####

        #np.where not working due to incompatible broadcast dimensions. needs fixing
        #tau = np.where(W==0, y1_control-y[W==0], y[W==1]-y0_treat)
        #return tau
    
    def fit_CATE(self, X, W, tau):
        self.tau0_base.fit(X[W==0], tau[W==0])
        self.tau1_base.fit(X[W==1], tau[W==1])

    def predict_CATE(self, X):
        tau0_preds = self.tau0_base.predict(X)
        tau1_preds = self.tau1_base.predict(X)
        
        tau_preds = g * tau0_preds + (1-g) * tau1_preds
        return tau_preds.flatten()

# Set root directory
base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

# Read in data
train = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_train.csv')
test = pd.read_csv(base_repo_dir / 'data' / 'simA' / 'samp1_test.csv')
X_train = train.drop(columns=['treatment', 'Y', 'tau'])
y_train = train['Y']
W_train = train['treatment']
X_test = test.drop(columns=['treatment', 'Y', 'tau'])

# intialize base learner
# bart = SklearnModel() # takes too long
rf = RegressionForest(honest=True, random_state=42)

# initialize metalearner
#keeping g static for now as 0.5 
g=np.full((1, len(X_test)), 0.5)
X_learner = x_learner(mu0_base=rf, mu1_base=rf, tau0_base=rf, tau1_base=rf, g=g)

# Fit treatment and response estimators mu0 and  mu1
X_learner.fit_t(X=X_train, W=W_train, y=y_train)

# Use mu1 to get imputed y for control group and mu0 to get imputed y for treatment group
#tau=X_learner.predict_impute(X=X_train, W=W_train, y=y_train)

#X_learner.fit_CATE(X=X_train, W=W_train, tau=tau)

#tau_preds = X_learner.predict_CATE(X=X_test)

# Calculate RMSE on test set
#rmse = np.sqrt(np.mean((tau_preds - test.tau)**2))

#print('tau_preds.shape: ', len(tau_preds)) # should be (1000,)
#print('RMSE: ', rmse)
