#!/usr/bin/env python3

'''
Estimating and evaluating bootstrap confidence intervals on simulated data
Alene Rhea, May 2021
'''

import argparse, os, pathlib, json
import numpy as np
import pandas as pd
from learners.py import fit_predict

def main(args):

    # Set root directory
    base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    # Read config  from JSON (?)
    learner_dict = json.load(args.config_file)

    # Read train and test from sample 1
    # Note: Using data repo structure with no train_size directory level
    train_filename = 'samp1_train.parquet'
    test_filename = 'samp1_train.parquet'
    train_full = pd.read_parquet(base_repo_dir / 'data' / args.sim / train_filename)
    test = pd.read_parquet(base_repo_dir / 'data' / args.sim / test_filename)

    # Sample training set from super-set of 300K
    if args.train_size < len(train_full):
        train = train_full.sample(n=args.train_size, replace=False, random_state=args.seed)
    else:
        train = train_full

    # Isolate treatment and control groups from the training set
    train_treat = train[train.treatment==1]
    train_ctrl = train[train.treatment==0]

    # Train B versions of the learner
    # CAN WE PARALLELIZE?
    for b in range(args.B):
        b_treat = train_treat.sample(frac=1, replace=True, random_state=b)
        b_ctrl = train_ctrl.sample(frac=1, replace=True, random_state=b)
        boot_df = pd.concat([b_treat, b_ctrl])

        # train model on boot_df; use learner_dict
        # predict on test set and save preds (to file if parallelized)

    # Make DF of predictions (length of test set x B)

    if args.order1_norm:
        # train another model on the original train; use these preds to center CI
        # get row-wise SD of test set preds; feed to Q func (with alpha) to get spread of CI
    if args.smooth:
        # center CI around mean of bootstrap preds (row-wise means)
        # Use covariance formula from SI 7 to get spread of CI
    if args.quantiles:
        # alpha/2 and 1-alpha/2 quantiles
        # same as percentile?
    if args.t:
        # https://mikelove.wordpress.com/2010/02/15/bootstrap-t/
    
    # Create CI df length of test set
    # Columns: true tau; for each CI method: CI bottom, CI top, true tau within CI (bool)
    # Save this DF to CSV so we can plot it

    # Also save DF of predictions? In case we want to re-use the data later?
    # e.g. to test different alpha level; test new bootstrap method

if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()

    # seed

    # sim

    # test size

    # config
    # hyperparams?

    args = parser.parse_args()

    # Call main routine
    main(args)