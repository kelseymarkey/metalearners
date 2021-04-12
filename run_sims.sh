#!/bin/bash

# This script generates a small amount of data:
# 30 samples of each simulation (A, B, C, and D) 
# Each has 3000 rows in train, 1000 in test

n_train=3000
n_test=1000
n_samples=30


# Array of error correlations to test
SIMS=('A' 'B' 'C' 'D')


for sim in ${SIMS[@]}
do
    echo "BEGIN SIMULATION $sim"

    # Create necessary directory
    mkdir -p "data/sim$sim/"

    # Initialize counter
    i=1
    while [ $i -le $n_samples ]
    do
    
        # Generate data
        Rscript generate_simulated_data.R --sim $sim --samp $i \
        --n_train $n_train --n_test $n_test 

        echo "     Finished generating $i/$n_samples samples of sim $sim "

        # Increase counter
        i=$(( $i + 1 ))

    done
done
