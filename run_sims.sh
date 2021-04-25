#!/bin/bash

# This script generates a training and test dataset for
# each simulation (A, B, C, D, E, and F) with 300K training rows
# (to be sampled from later), and 100K test rows.
# Will be run in an sbatch array SLURM job on the Greene cluster,
# to produce 30 samples of each.

n_train=300000
n_test=100000
n_samples=30

# Array of simulations to generate
SIMS=('A' 'B' 'C' 'D' 'E' 'F')

for sim in ${SIMS[@]}
do
	echo "BEGIN SIMULATION $sim"

	# Generate data
		Rscript generate_simulated_data.R --sim $sim --samp $SLURM_ARRAY_TASK_ID \
		--n_train $n_train --n_test $n_test

		echo "     Finished generating $SLURM_ARRAY_TASK_ID/$n_samples samples of sim $sim "
done
