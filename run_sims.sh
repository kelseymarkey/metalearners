#!/bin/bash

# This script generates a training and test dataset for each
# training set size in n_train. Test set sizes are all 100,000 rows.
# 30 samples of each simulation (A, B, C, D, E, and F) 

n_train=300000
n_test=100000
n_samples=2


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

	    # Initialize counter
	    i=1
	    while [ $i -le $n_samples ]
	    do
	        # Generate data
	        Rscript generate_simulated_data.R --sim $sim --samp $i \
	        --n_train $train_size --n_test $n_test

	        echo "     Finished generating $i/$n_samples samples of sim $sim "

	        # Increase counter
	        i=$(( $i + 1 ))

	    done
	done
done
