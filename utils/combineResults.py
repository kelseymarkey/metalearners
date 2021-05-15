#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pathlib, os

base_repo_dir = pathlib.Path(os.path.realpath(__file__)).parents[1]

# Lists of all dimensions to loop through to get all CIs we calculated
configs_long = ['iw_g_rfc_default', 'rf_g_rfc_default']
configs_short = ['iw', 'rf']
sims = ['E', 'F']
Bs = [100, 1000, 5000, 10000]
metas = ['S', 'T', 'X']
ci_types = ['paper', 'percentile', 'basic', 'normal']

# Initialize DF to hold all CI results together
results_cols = ['ci_type', 'sim', 'meta', 'base', 'B', 'coverage', 'mean_length', 'mse_avg', 'mse_samp1']
res = pd.DataFrame(columns=results_cols)

# Loop through sims, metalearners, baseleaners, and B values to read all results
for config in configs_long:
    # Read in mse results
    mse_avg = pd.read_csv(base_repo_dir / 'results/mse/results_{}_10.csv'.format(config))
    mse_samp = pd.read_csv(base_repo_dir / 'results/mse/results_full_{}_10.csv'.format(config))
    # Only 20K is relevant to CI results; and 1st sample
    mse_avg = mse_avg[mse_avg.n==20000]
    mse_samp = mse_samp[(mse_samp.n==20000)&(mse_samp.trial==0)]
    for  sim in sims:
        for meta in metas:
            for B in Bs:
                try:

                    # Read simple CI result file
                    new_res = pd.read_csv(base_repo_dir / 'results' /'ci'/'simple'/\
                                        '{}_{}_sim{}_tsize20000_B{}_simple.csv'.format(config, meta, sim, B))

                    # Add data about which run this came from
                    new_res['meta']=meta
                    new_res['base']=config.split('_')[0]
                    new_res['sim']=sim
                    new_res['B']=B
                    
                    # Add 20K MSE data
                    new_res['mse_avg']=mse_avg.loc[mse_avg.simulation=='sim'+sim, meta+'_mse'].values[0]
                    new_res['mse_samp1']=mse_samp.loc[mse_samp.simulation=='sim'+sim, meta+'_mse'].values[0]
                    
                    #Append to big results df
                    res = pd.concat([res,new_res])
                except:
                    # exception for CI runs that aren't done yet
                    continue

# remove outdated CI type from df
res=res[res.ci_type!='order1norm']

# Map cleaner shorthand names
ci_dict = {'paper':'ACIC', 'normal':'Normal', 'percentile':'Percentile', 
           'basic': 'Basic', 'studentized': 'Studentized'}
base_dict = {'rf': 'RF', 'iw':'LR-IW'}
res['ci_type']=res['ci_type'].map(ci_dict)
res['base'] = res['base'].map(base_dict)

# Save to CSV
res.to_csv(base_repo_dir / 'results/all_ci_mse_simple.csv', index=False)