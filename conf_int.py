#!/usr/bin/env python3

'''
Estimating and evaluating bootstrap confidence intervals on simulated data
Alene Rhea, May 2021

Usage example: 
python3 conf_int.py --meta S --sim E --train_size 5000 --B 5 \
--base_learner_filename base_learners_iw_g_logreg.json --hp_substr default 

python conf_int.py --meta S --sim E --train_size 20000 --B 10000 \
--base_learner_filename base_learners_iw_g_rfc.json --hp_substr default \
--percentile --normal --basic --paper \
--results_file iw_g_rfc_default_S_simE_tsize20000_B10000_full.parquet 
'''

import argparse, os, pathlib, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from functools import partial
import re
from learners import fit_predict_ci
import utils.configClass
import utils.metaClass
from utils.utils import strat_sample, config_from_json

def get_ci (boot_preds, alpha, ci_type, M=None,
            train=None, test=None, config=None):
    '''
    boot_preds is an array
    '''

    # 1st order normal approximation
    if ci_type=='normal':
        #https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/1365-2656.12382

        # center around mean of bootstrap preds
        center_preds = np.mean(boot_preds, axis=1)

        # Get Z-score for given alpha value
        z = norm.ppf(1-alpha/2)
        
        # get row-wise SD of bootstrap preds on test set
        sigma = np.std(boot_preds, axis=1)

        # get radius of CI
        radius = z*sigma

        # CI is symmetric around means
        lower = center_preds - radius
        upper = center_preds + radius

     # Equivalent to SI algorithm 6
    if ci_type=='paper':
        # train another model on the original train set
        center_preds = fit_predict_ci(train=train, test=test, 
                                      config=config)

        # get row-wise SD of bootstrap preds on test set
        sigma = np.std(boot_preds, axis=1)

        # CI is asymmetric around preds on original test set
        lower = center_preds - sigma*np.quantile(boot_preds, q=alpha/2, axis=1)
        upper = center_preds + sigma*np.quantile(boot_preds, q=(1-alpha)/2, axis=1)
    
    # Simple percentile method
    elif ci_type=='percentile':
        # alpha/2 and 1-alpha/2 percentiles
        lower = np.quantile(boot_preds, q=alpha/2, axis=1)
        upper = np.quantile(boot_preds, q=1-(alpha/2), axis=1)

    # "Basic" method, also called reverse percentile
    elif ci_type=='basic':
        # train another model on the original train set
        center_preds = fit_predict_ci(train=train, test=test, 
                                      config=config)

        # alpha/2 and 1-alpha/2 percentiles
        lower = 2*center_preds - np.quantile(boot_preds, q=1-(alpha/2), axis=1)
        upper = 2*center_preds - np.quantile(boot_preds, q=alpha/2, axis=1)

    # "Studentized" method, also called bootstrap-t
    elif ci_type=='studentized':
        # https://mikelove.wordpress.com/2010/02/15/bootstrap-t/
        # https://www.stat.cmu.edu/~ryantibs/advmethods/notes/bootstrap.pdf

        # train another model on the original train set
        center_preds = fit_predict_ci(train=train, test=test, 
                                      config=config)
        B = boot_preds.shape[1]

        # Initialize array to hold "pivotal" values
        all_t = np.zeros((len(test), B))
        for b in range(B):
            # Reconstruct each bootstrap sample
            # Assumes boot_preds starts with 'b1_tau' and uses continuous seeds
            boot_df = strat_sample(train, n=len(train), replace=True, seed=b)

            # Initialize array to hold all 2nd-level bootstrap predictions
            all_m_preds = np.zeros((len(test), M))
            for m in range(M):
                # Get a new boostrap sample and re-estimate CATEs
                boot2 = strat_sample(boot_df, n=len(boot_df), replace=True, seed=m)
                all_m_preds[:,m] = fit_predict_ci(train=boot2, test=test,
                                                    config=config) 

            # get row-wise SD of 2nd-level bootstrap preds
            sigma_b = np.std(all_m_preds, axis=1)

            # calculate "pivotal quantity" (allegedly t-distributed)
            t = (boot_preds[:,b] - center_preds)/sigma_b
            all_t[:,b] = t

        # Get standard deviation of 1st-level bootstrap preds
        sigma = np.std(boot_preds, axis=1)

        # asymmetric around new preds, based on pivot*sd of 1st level
        lower = center_preds - sigma*np.quantile(all_t, q=1-(alpha/2), axis=1)
        upper = center_preds - sigma*np.quantile(all_t, q=alpha/2, axis=1)

    return (lower, upper)


def main(args):

    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    # Array of string-names of conf interval types
    ci_types = np.array(['paper', 'percentile', 'basic', 'normal', 'studentized'])

    # Include only those passed as true
    ci_types = ci_types[[args.paper, args.percentile, args.basic, args.normal, args.studentized]]


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

    # Set flag to save full results file
    save_full = True

    # Generate bootstrap predictions if results file not already passed
    if not args.results_file:

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
        results = pd.DataFrame(all_preds, columns=['b{}_tau'.format(x+1) for x in np.arange(args.B)])

        # Save true tau
        results['true_tau'] = test['tau']

        # Initialize dataframe to hold simple coverage results
        coverage = pd.DataFrame(columns=['ci_type', 'coverage', 'mean_length'])

    # results file already passed
    else:
        # read in results
        results = pd.read_parquet(base_repo_dir/'results/ci/full'/args.results_file)

        # reconstruct all_preds numpy array (include only b#_tau columns)
        boot_cols = [x for x in results.columns if '_tau' in x and 'true' not in x]
        all_preds = results[boot_cols].to_numpy()

        # read in corresponding simple results
        coverage = pd.read_csv(base_repo_dir/'results/ci/simple'/args.results_file.replace('full', 'simple').replace('parquet', 'csv'))

        if args.B<len(boot_cols):
            # Limit all_preds to the first B columns
            all_preds = all_preds[:,:args.B]

            # Clear old CI information and make smaller version of results df
            results = pd.DataFrame(all_preds, columns=['b{}_tau'.format(x+1) for x in np.arange(args.B)])
            
            # Save true tau
            results['true_tau'] = test['tau']

            # Initialize dataframe to hold simple coverage results for this B value
            coverage = pd.DataFrame(columns=['ci_type', 'coverage', 'mean_length'])

            # Do not save full version of results if bigger version of B exists
            save_full = False
        
    for c in ci_types:

        if c in coverage.ci_type.values:
            print('Skipping '+c+' CIs. Already exists in simple results file.')
            continue

        # Get CI lower and upper bounds
        lower, upper = get_ci(boot_preds = all_preds, alpha=args.alpha, 
                              ci_type = c, M=args.M,
                              train=train, test=test, config=config)
        results[c+'_lower'] = lower
        results[c+'_upper'] = upper

        # Add Boolean coverage column
        results[c+'_cover'] = ( (results[c+'_lower'] <= results['true_tau']) \
                               &(results[c+'_upper'] >= results['true_tau']))
        
        # Save length of each CI
        results[c+'_length'] = results[c+'_upper'] - results[c+'_lower']

        # Update coverage df
        coverage = coverage.append({'ci_type':c, 
                                    'coverage':np.sum(results[c+'_cover'])/len(test),
                                    'mean_length':np.mean(results[c+'_length'])},
                                    ignore_index=True)
        # print progress update
        print(c + ' CIs completed')

    # Save condensed results file
    simple_dir = os.path.join(base_repo_dir, 'results', 'ci', 'simple')
    if not os.path.exists(simple_dir):
            os.makedirs(simple_dir)
    coverage.to_csv(os.path.join(simple_dir, filename_str+'_simple.csv'), index=False)

    if save_full:
        # Save full results file to parquet (including all predicitions)
        full_dir = os.path.join(base_repo_dir, 'results', 'ci', 'full')
        if not os.path.exists(full_dir):
                os.makedirs(full_dir)
        results.to_parquet(os.path.join(full_dir, filename_str+'_full.parquet'))




if __name__ == "__main__":

    #Command line arguments
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--B", type=int, 
                        help='Number of bootstrap samples to build metalearners with. Can pass new value of B if bootstrap_preds already exists.')
    parser.add_argument("--sim", type=str, 
                        help='Which simulation to calculate CIs on')
    parser.add_argument("--meta", type=str,
                        help="Which metalearner to use. Must be one of 'X', 'T', or 'S'")
    parser.add_argument("--base_learner_filename", type=str,
                        help='Name of base learner file to use. Should be of form base_learners_XX.json ' +
                        'and reside in configurations/base_learners')

    # Optional arguments
    parser.add_argument("--hp_substr", type=str, default='default',
                        help='The naming convention for the hyperparameter files that should be used. For ' +
                        'example if user wishes to use rf_t_default.json/rf_s_default.json/etc. then the string ' +
                        'passed should be default.')
    parser.add_argument("--train_seed", type=int, default=42,
                        help='Seed to use to sample training set from 300K superset')
    parser.add_argument("--train_size", type=int, default=20000,
                        help='Number of observations to sample to create training set')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help='Significance level for confidence intervals. Must be in (0,1). Default of 0.05 corresponds to 95% CI.')
    parser.add_argument("--M", type=int, default=None,
                        help="Number of second-level bootstrap samples to use in studentized CI method.")
    parser.add_argument("--normal", action='store_true',
                        help="Boolean flag to construct CIs with 1st order normal approximation. Will default to true if no other CI types are selected.")
    parser.add_argument("--paper", action='store_true',
                        help="Boolean flag to construct CIs with asymmetrical method used in paper.")
    parser.add_argument("--basic", action='store_true',
                        help="Boolean flag to construct CIs with the \"basic\" method.")
    parser.add_argument("--percentile", action='store_true',
                        help="Boolean flag to construct CIs with the percentile method.")
    parser.add_argument("--studentized", action='store_true',
                        help="Boolean flag to construct CIs with the \"t\" method.")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Filename of results file saved in results/ci/full. For use when bootstrap predictions have already been calculated.")
    
    args = parser.parse_args()

    # Call main routine
    main(args)