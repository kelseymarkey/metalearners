#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression

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

    def fit(self, X, W, y, g=None):

        if type(g)!=type(None):
            g0 = g[W==0]
            g1 = g[W==1]

        # Call fit methods on each base learner
        if isinstance(self.mu0_base, LinearRegression):
            # fit with IPW sample weights (unbiased for full data)
            self.mu0_base.fit(X[W==0], y[W==0], sample_weight=1/(1-g0))
        else:
            self.mu0_base.fit(X[W==0], y[W==0])
        if isinstance(self.mu1_base, LinearRegression):
            # fit with IPW sample weights (unbiased for full data)
            self.mu1_base.fit(X[W==1], y[W==1], sample_weight=1/g1)
        else:
            self.mu1_base.fit(X[W==1], y[W==1])
            
    def predict(self, X, export_preds=False):
        y1_preds = self.mu1_base.predict(X)
        y0_preds = self.mu0_base.predict(X)
        tau_preds = y1_preds - y0_preds

        if export_preds:
            export_df = pd.DataFrame()
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

    def fit(self, X_W, y, *argv):
        # fit on full data, so unbiased
        self.mu_base.fit(X_W, y)

    def predict(self, X, export_preds=False):
        X_W_1 = X.copy(deep=True)
        X_W_1["W"] = 1
        y1_preds = self.mu_base.predict(X_W_1)

        X_W_0 = X.copy(deep=True)
        X_W_0["W"] = 0
        y0_preds = self.mu_base.predict(X_W_0)

        tau_preds = y1_preds - y0_preds

        if export_preds:
            export_df = pd.DataFrame()
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

    def fit(self, X, W, y, g=None, export_preds=False):

        if type(g)!=type(None):
            g0 = g[W==0]
            g1 = g[W==1]

        # Call fit method
        if isinstance(self.mu0_base, LinearRegression):
            # fit with IW sample weights (unbiased for incomplete data)
            self.mu0_base.fit(X[W==0], y[W==0], sample_weight=g0/(1-g0))
        else:
            self.mu0_base.fit(X[W==0], y[W==0])

        if isinstance(self.mu1_base, LinearRegression):
            # fit with IW sample weights (unbiased for incomplete data)
            self.mu1_base.fit(X[W==1], y[W==1], sample_weight=(1-g1)/g1)
        else:
            self.mu1_base.fit(X[W==1], y[W==1])
    
        #Impute y0 for treated group using mu0
        y0_treat=self.mu0_base.predict(X[W==1]).flatten()
        imputed_TE_treatment = y[W==1] - y0_treat

        #Impute y1 for control group using mu1
        y1_control=self.mu1_base.predict(X[W==0]).flatten()
        imputed_TE_control = y1_control - y[W==0]

        #Fit tau0: CATE estimate fit to the control group
        if isinstance(self.tau0_base, LinearRegression):
            # fit with IPW sample weights (unbiased for full data)
            self.tau0_base.fit(X[W==0], imputed_TE_control, sample_weight=1/(1-g0))
        else:
            self.tau0_base.fit(X[W==0], imputed_TE_control)

        #Fit tau1: CATE estimate fit to the treatment group
        if isinstance(self.tau1_base, LinearRegression):
            # fit with IPW sample weights (unbiased for full data)
            self.tau1_base.fit(X[W==1], imputed_TE_treatment, sample_weight=1/g1)
        else:
            self.tau1_base.fit(X[W==1], imputed_TE_treatment)

        if export_preds:
            # Export y0_treat and y1_control predictions on training set
            y0_treat_df = pd.DataFrame()
            y0_treat_df["y0_treat_preds"] = y0_treat
            y0_treat_df["y"] = y[W==1]
            y0_treat_df["W"] = 1
            y0_treat_df['pscore_preds'] = g1

            y1_control_df = pd.DataFrame()
            y1_control_df["y1_control_preds"] = y1_control
            y1_control_df["y"] = y[W==0]
            y1_control_df["W"] = 0
            y1_control_df['pscore_preds'] = g0

            export_df_train = y0_treat_df.append(y1_control_df, ignore_index=True, sort=False)
        else:
            export_df_train = None
        return export_df_train

    def predict(self, X, g, export_preds=False):
        '''
        g : weight vector that should be length of the test set
        '''
        tau0_preds = self.tau0_base.predict(X).flatten()
        tau1_preds = self.tau1_base.predict(X).flatten()
        tau_preds = (g.T * tau0_preds) + ((1-g).T*tau1_preds)

        if export_preds:
            export_df_test = pd.DataFrame({'tau_preds': tau_preds})
            export_df_test['pscore_preds'] = g
            return export_df_test
        else:
            return tau_preds.flatten()
