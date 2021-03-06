#!/bin/bash

# This script generates a training and test dataset for each
# simulation in SIMS. Training set sizes are all 300,000 rows,
# test set sizes are all 100,000 rows. 30 samples of each
# simulation (A, B, C, D, E, and F) are created.

n_train=300000
n_test=100000
n_samples=30


# Array of simulations to generate
SIMS=('A' 'B' 'C' 'D' 'E' 'F')


for train_size in ${n_train[@]}
do
	# Create necessary directory
	mkdir -p "data/"

	for sim in ${SIMS[@]}
	do
		echo "BEGIN SIMULATION $sim"

	    # Create necessary directory
	    mkdir -p "data/sim${sim}"

		# Generate 1st sample and save extra columns
		Rscript generate_simulated_data.R --sim $sim --samp 1 \
		--n_train $train_size --n_test $n_test --extra_cols

		echo "     Finished generating 1/$n_samples samples of sim $sim "

	    # Initialize counter
	    i=2
	    while [ $i -le $n_samples ]
	    do
	        # Generate data for all other samples (no extra cols)
	        Rscript generate_simulated_data.R --sim $sim --samp $i \
	        --n_train $train_size --n_test $n_test

	        echo "     Finished generating $i/$n_samples samples of sim $sim "

	        # Increase counter
	        i=$(( $i + 1 ))

	    done
	done
done
