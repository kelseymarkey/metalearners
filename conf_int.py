#!/usr/bin/env python3

'''
Estimating and evaluating bootstrap confidence intervals on simulated data
Alene Rhea, May 2021

Usage example: 
python3 conf_int.py --meta S --sim E --train_size 5000 --B 5 \
--base_learner_filename base_learners_iw_g_logreg.json --hp_substr default 

To install varname:
pip install -U varname
'''

import argparse, os, pathlib, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from functools import partial
from varname import nameof
import re
from learners import fit_predict_ci
import utils.configClass
import utils.metaClass
from utils.utils import strat_sample, config_from_json

def get_ci (boot_preds, alpha, ci_type, 
            train=None, test=None, config=None):

    # Get Z-score for given alpha value
    z = norm.ppf(1-alpha/2)
    
    # Equivalent to SI algorithm 6
    if ci_type=='order1norm':
        
        # train another model on the original train set
        center_preds = fit_predict_ci(train=train, test=test, 
                                   config=config)

        # get row-wise SD of bootstrap preds on test set
        sigma = np.std(boot_preds, axis=1)

        # get radius of CI
        radius = z*sigma

        # CI is symmetric around preds on original training set
        lower = center_preds - radius
        upper = center_preds + radius

    # Equivalent to SI algorithm 7
    elif ci_type=='smooth':

        # center CI around mean of bootstrap preds
        center_preds = np.mean(boot_preds, axis=1)

        # Use covariance formula from SI 7 to get spread of CI
        lower = None
        upper = None
    
    elif ci_type=='quantile':
        # alpha/2 and 1-alpha/2 quantiles
        lower = None
        upper = None

    elif ci_type=='t':
        # https://mikelove.wordpress.com/2010/02/15/bootstrap-t/
        lower = None
        upper = None

    return (lower, upper)

def main(args):

    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    # Read learner config from JSON
    meta_base_dict = json.load(open(base_repo_dir /\
                                    'configurations/base_learners'/\
                                     args.base_learner_filename))

    # Read hyperparameters file
    hp = json.load(open(base_repo_dir / 'configurations' / 'hyperparameters' / 'rf_{}_{}.json'.format(args.meta.lower(),
                                                                                                     args.hp_substr)))

    # create config object
    # Needs to be updated for RandomForestClassifier params (for g)
    config = config_from_json(meta=args.meta, sim=args.sim, 
                                meta_base_dict=meta_base_dict, 
                                hyperparams=hp)

    # Read train and test from sample 1
    # Note: Using data repo structure with no train_size directory level
    train_filename = 'samp1_train.parquet'
    test_filename = 'samp1_test.parquet'
    train_full = pd.read_parquet(base_repo_dir/'data'/\
                                 'sim{}'.format(args.sim)/train_filename)
    test = pd.read_parquet(base_repo_dir/'data'/\
                                 'sim{}'.format(args.sim)/test_filename)

    # Create filename substring for use in results files
    bl_substr = re.search('base_learners_(.*?).json', args.base_learner_filename).group(1)
    filename_str = '{}_{}_{}_sim{}_tsize{}_B{}'\
                    .format(bl_substr, args.hp_substr, args.meta, args.sim, args.train_size, args.B)

    # Sample training set from super-set of 300K
    if args.train_size < len(train_full):
        train = strat_sample(train_full, n=args.train_size)
    else:
        train = train_full

    # Initialize array to hold all bootstrap predictions
    all_preds = np.zeros((len(test), args.B))

    # Train B versions of the learner
    # CAN WE PARALLELIZE?
    for b in range(args.B):

        # Sample new bootstrap df (with replacement)
        boot_df = strat_sample(train, n=len(train), replace=True, seed=b)

        # train model on bootstrapped df
        these_preds = fit_predict_ci(train=boot_df, test=test,
                                  config=config) 

        # If parallelized, should write each array to file?
        # Or collect results with fancy array method?
        all_preds[:,b] = these_preds

        print('{}/{} bootstrap samples complete'.format(b+1, args.B))

    # Save all_preds in case we want to do more calculation later
    results = pd.DataFrame(all_preds, 
                columns=['b{}_tau'.format(x+1) for x in np.arange(args.B)])

    # Save true tau
    results['true_tau'] = test['tau']

    # Array of string-names of conf interval types
    ci_types = np.array([nameof(args.order1norm), nameof(args.smooth), 
                nameof(args.quantile), nameof(args.t)])

    # Include only those passed as true
    ci_types = ci_types[[args.order1norm, args.smooth, 
                         args.quantile, args.t]]

    # Initialize dataframe to hold simple coverage results
    coverage = pd.DataFrame(columns=['coverage', 'mean_length'],
                            index=ci_types)

    for c in ci_types:

        # Get CI lower and upper bounds
        lower, upper = get_ci(boot_preds = all_preds, alpha =args.alpha, 
                              ci_type = c, 
                              train=train, test=test, config=config) # for order1norm
        results[c+'_lower'] = lower
        results[c+'_upper'] = upper

        # Add Boolean coverage column
        results[c+'_cover'] = ( (results[c+'_lower'] <= results['true_tau']) \
                               &(results[c+'_upper'] >= results['true_tau']))
        
        # Save length of each CI
        results[c+'_length'] = results[c+'_upper'] - results[c+'_lower']

        # Save proportion of covered points
        coverage.loc[c, 'coverage'] = np.sum(results[c+'_cover'])/len(test)

        # Save mean CI length
        coverage.loc[c, 'mean_length'] = np.mean(results[c+'_length'])

    # Save full results file to parquet (including all predicitions)
    full_dir = os.path.join(base_repo_dir, 'results', 'ci', 'full')
    if not os.path.exists(full_dir):
            os.makedirs(full_dir)
    results.to_parquet(os.path.join(full_dir, filename_str+'_full.parquet'))

    # Save condensed results file
    simple_dir = os.path.join(base_repo_dir, 'results', 'ci', 'simple')
    if not os.path.exists(simple_dir):
            os.makedirs(simple_dir)
    coverage.to_csv(os.path.join(simple_dir, filename_str+'_simple.csv'), index=False)


if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--sim", type=str, 
                        help='Which simulation to calculate CIs on')
    parser.add_argument("--meta", type=str,
                        help="Which metalearner to use. Must be one of 'X', 'T', or 'S'")
    parser.add_argument("--outfile", type=str, 
                        help='String to insert into outfile names. Should uniquely identify the type of learner & config being tested, as well as the simulation.')
    parser.add_argument("--base_learner_filename", type=str, default='base_learners.json',
                        help='Name of base learner file to use. Should be of form base_learners_XX.json ' +
                        'and reside in configurations/base_learners')
    parser.add_argument("--hp_substr", type=str, default='default',
                        help='The naming convention for the hyperparameter files that should be used. For ' +
                        'example if user wishes to use rf_t_default.json/rf_s_default.json/etc. then the string ' +
                        'passed should be default.')

    # Optional arguments
    parser.add_argument("--train_seed", type=int, default=42,
                        help='Seed to use to sample training set from 300K superset')
    parser.add_argument("--train_size", type=int, default=20000,
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
    
    args = parser.parse_args()

    # Calculate CIs via order1norm if no other methods selected
    if (not args.smooth) and ((not args.quantile ) and (not args.t)):
        args.order1norm = True

    # Call main routine
    main(args)