#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem=4GB
#SBATCH --job-name=metalearners_data_gen
##SBATCH --output=$SCRATCH/$USER/slurm_%j_%a.out

module purge
module load r/gcc/4.0.4

# This script generates a training and test dataset for
# each simulation (A, B, C, D, E, and F) with 300K training rows
# (to be sampled from later), and 100K test rows.
# Will be run in an sbatch array SLURM job on the Greene cluster,
# to produce 30 samples of each.
# To run on command line in Greene cluster:
#     sbatch --array=1-30 run_sims.sh

n_train=300000
n_test=100000
n_samples=30

mkdir "/home/$USER/R/4.0.4"

R --no-save -q -e 'install.packages("arrow", lib="/home/$USER/R/4.0.4", repos="https://cran.r-project.org”)'
R --no-save -q -e 'install.packages("argparse", lib="/home/$USER/R/4.0.4", repos="https://cran.r-project.org”)'

# Array of simulations to generate
SIMS=('A' 'B' 'C' 'D' 'E' 'F')

for sim in ${SIMS[@]}
do
	echo "BEGIN SIMULATION $sim"

	# Generate data
	Rscript generate_simulated_data.R --sim $sim --samp $SLURM_ARRAY_TASK_ID \
	--n_train $n_train --n_test $n_test --user $USER

	echo "     Finished generating $SLURM_ARRAY_TASK_ID/$n_samples samples of sim $sim "
done
