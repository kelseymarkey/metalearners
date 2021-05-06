#!/usr/bin/env python3

'''
Estimating and evaluating bootstrap confidence intervals on simulated data
Alene Rhea, May 2021
'''

import argparse, os, pathlib, json
import numpy as np
import pandas as pd
from scipy.stats import norm
#from learners.py import fit_predict

def main(args):

    #fake test set & args, to test functionality
    # test = pd.DataFrame({'tau': np.random.random(10000)})
    # class arg:
    #     def __init__(self, B, sim, alpha, outfile):
    #         self.B = B
    #         self.sim = sim
    #         self.alpha = alpha
    #         self.order1norm = True
    #         self.smooth = False
    #         self.quantile = False
    #         self.t = False
    #         self.outfile = outfile
    # args = arg(3, 'A', 0.05, 'testing')

    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    # Read config from JSON (?)
    learner_dict = json.load(args.config_file)

    # Read hyperparams?

    # Read train and test from sample 1
    # Note: Using data repo structure with no train_size directory level
    train_filename = 'samp1_train.parquet'
    test_filename = 'samp1_test.parquet'
    train_full = pd.read_parquet(base_repo_dir / 'data' / args.sim / train_filename)
    test = pd.read_parquet(base_repo_dir / 'data' / args.sim / test_filename)

    # Sample training set from super-set of 300K
    if args.train_size < len(train_full):
        train = train_full.sample(n=args.train_size, replace=False, 
                                  random_state=args.train_seed)
    else:
        train = train_full

    # Isolate treatment and control groups from the training set
    train_treat = train[train.treatment==1]
    train_ctrl = train[train.treatment==0]

    # Initialize array to hold all bootstrap predictions
    all_preds = np.zeros((len(test), args.B))

    # Train B versions of the learner
    # CAN WE PARALLELIZE?
    for b in range(args.B):

        # Sample treatment and control seperately and then combine
        b_treat = train_treat.sample(frac=1, replace=True, random_state=b)
        b_ctrl = train_ctrl.sample(frac=1, replace=True, random_state=b)
        boot_df = pd.concat([b_treat, b_ctrl])

        # train model on bootstrapped df
        # these_preds = fit_predict(boot_df) 

        # generate fake preds to test CI calculation
        # these_preds = np.random.random(size=len(test))

        # If parallelized, should write each array to file?
        # Or collect results with fancy array method?
        all_preds[:,b] = these_preds

    # Save all_preds in case we want to do more calculation later
    results = pd.DataFrame(all_preds, 
                columns=['b{}'.format(x) for x in np.arange(args.B)])

    # Save true tau
    results['true_tau'] = test['tau']

    # Get signif level for given alpha value
    # (Uses normal distribution)
    signif_level = norm.ppf(1-args.alpha/2)

    if args.order1norm:

        # train another model on the original train
        # norm_center_preds = fit_predict(train)

        # generate fake preds to test CI calculation
        # norm_center_preds = np.random.random(size=len(test))

        #get row-wise SD of test set preds
        norm_sigma = np.std(all_preds, axis=1)

        # CI is symmetric around preds on original training set
        results['order1norm_lower'] = norm_center_preds \
                                            - signif_level*norm_sigma
        results['order1norm_upper'] = norm_center_preds \
                                            + signif_level*norm_sigma

    if args.smooth:

        # center CI around mean of bootstrap preds
        smooth_center_preds = np.mean(all_preds, axis=1)

        # Use covariance formula from SI 7 to get spread of CI

        #results['smooth_lower'] = smooth_center_preds - ...
        #results['smooth_upper'] = smooth_center_preds + ...

    if args.quantiles:
        # alpha/2 and 1-alpha/2 quantiles
        # same as percentile?
        pass
    if args.t:
        # https://mikelove.wordpress.com/2010/02/15/bootstrap-t/
        pass

    # List of string-names for the CI types calculated
    ci_types = [x.split('_')[0] for x in results.columns\
                    if 'lower' in str(x)]

    # Initialize dataframe to hold simple coverage results
    coverage = pd.DataFrame(columns=['coverage', 'mean_length'],
                            index=ci_types)

    for c in ci_types:
        # Add Boolean coverage column for each CI type
        results[c+'_cover'] = ( (results[c+'_lower'] <= results['true_tau']) \
                               &(results[c+'_upper'] >= results['true_tau']))
        
        # Save length of each CI
        results[c+'_length'] = results[c+'_upper'] - results[c+'_lower']

        # Save proportion of covered points
        coverage.loc[c, 'coverage'] = np.sum(results[c+'_cover'])/len(test)

        # Save mean CI length
        coverage.loc[c, 'mean_length'] = np.mean(results[c+'_length'])

    # Save big results file (including all predicitions)
    results.to_csv('ci_results_'+args.outfile_name+'_full.csv')

    # Save condensed results file
    coverage.to_csv('ci_results_'+args.outfile_name+'_simple.csv')

    pass

    

if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser()

    # Required argument
    parser.add_argument("--sim", type=str, 
                        help='Which simulation to calculate CIs on')
    parser.add_argument("--outfile", type=str, 
                        help='String to insert into outfile names. Should uniquely identify the type of learner & config being tested, as well as the simulation.')
    
    # Optional arguments
    parser.add_argument("--train_seed", type=int, default=42,
                        help='Seed to use to sample training set from 300K superset')
    parser.add_argument("--train_size", type=int, default=10000,
                        help='Number of observations to sample to create training set')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help='Significance level for confidence intervals. Must be in (0,1). Default of 0.05 corresponds to 95% CI.')
    parser.add_argument("--B", type=int, default=10000,
                        help='Number of bootstrap samples to build metalearners with.')
    parser.add_argument("--order1norm", action='store_true',
                        help="Boolean flag to construct CIs with 1st order normal approximation. Will default to true if no other CI types are selected.")
    parser.add_argument("--smooth", action='store_true',
                        help="Boolean flag to construct CIs with the \"smooth\" method.")
    parser.add_argument("--quantile", action='store_true',
                        help="Boolean flag to construct CIs with the quantile method.")
    parser.add_argument("--t", action='store_true',
                        help="Boolean flag to construct CIs with the \"t\" method.")
    
    # Add config file
    # Add hyperparams file

    args = parser.parse_args()

    # Calculate CIs via order1norm if no other methods selected
    if (not args.smooth) and ((not args.quantile ) and (not args.t)):
        args.order1norm = True

    # Call main routine
    main(args)