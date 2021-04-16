#!/usr/bin/env python3

# from bartpy.sklearnmodel import SklearnModel
from econml.grf import RegressionForest
from sklearn import clone
import os, pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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
train = pd.read_parquet(base_repo_dir / 'data' / 'simA' / 'samp1_train.parquet')
test = pd.read_parquet(base_repo_dir / 'data' / 'simA' / 'samp1_test.parquet')
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

print('T Learner tau_preds.shape: ', len(tau_preds)) # should be (1000,)
print('T Learner RMSE: ', rmse)


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

# Fit S-learner
X_W = pd.concat([X_train, W_train], axis=1)
S.fit(X_W=X_W, y=y_train)

# Predict test-set CATEs
tau_preds = S.predict(X=X_test)

# Calculate RMSE on test set
rmse = np.sqrt(np.mean((tau_preds - test.tau)**2))

print('S Learner tau_preds.shape: ', len(tau_preds)) # should be (1000,)
print('S Learner RMSE: ', rmse)

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

    def fit(self, X, W, y, fit_g):
        '''
        fit_g : boolean indicator if g should be fit to training data. if false, g must be passed explicitlly to x_learner.predict()
        '''
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

        if fit_g:
            print('X Learner with g fitted')
            #predict propensities from empirical data
            self.g_fit = LogisticRegression(fit_intercept=True, max_iter=2000).fit(
            X=X, y=W)
        
        else:
            print('X Learner with true propensities')
            self.g_fit = None


    def predict(self, X, g):
        '''
        g : weight vector that should be length of the test set. can be passed as None if g was fit to data
        '''
        tau0_preds = self.tau0_base.predict(X).flatten()
        tau1_preds = self.tau1_base.predict(X).flatten()
    
    
        if self.g_fit == None:
            tau_preds = (g.T * tau0_preds) + ((1-g).T*tau1_preds)
        else:
            g_preds = self.g_fit.predict_proba(X)[:, 1]
            tau_preds = (g_preds.T * tau0_preds) + ((1-g_preds).T*tau1_preds)

        # idea: allow g to be either be a vector or function?
        # if function: think about sklearn inputs (.predict) vs lamba functions (g(x))
        # if g_type == 'Function':
        #    g_preds = g(X)
        #    tau_preds = (g_preds.T * tau0_preds) + ((1-g_preds).T*tau1_preds)

        return tau_preds.flatten()

# intialize base learner
# bart = SklearnModel() 
rf = RegressionForest(honest=True, random_state=42)

# initialize metalearner
X_learner = x_learner(mu0_base=rf, mu1_base=rf, tau0_base=rf, tau1_base=rf)

# initiliaze g
g = np.full((len(X_test)), 0.01)
#g = LogisticRegression(fit_intercept=True, max_iter=2000).fit(
#       X=X_train, y=W_train)

# Fit treatment and response estimators mu0 and  mu1
X_learner.fit(X=X_train, W=W_train, y=y_train, fit_g=False)

# Predict test-set CATEs
tau_preds = X_learner.predict(X=X_test, g=g)

# Calculate RMSE on test set
rmse = np.sqrt(np.mean((tau_preds - test.tau)**2))

print('X Learner tau_preds.shape: ', len(tau_preds)) # should be (1000,)
print('X Learner RMSE: ', rmse)
