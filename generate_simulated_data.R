# Generate data for specific simulations
# Simulations A, B, C, and D recreate synthetic datasets from Kunzel et al
library("arrow", warn.conflicts = FALSE)

# Valid simulations ready to be run
working_sims <- c('A', 'B', 'C', 'D', 'E', 'F')

# Get arguments from command line
library("argparse")
parser <- ArgumentParser(description='Generate synthetic data')
parser$add_argument('--sim', type="character",
                    help=paste("Which simulation to run. Must be one of: (",
                               paste(working_sims, collapse=', '), ")", sep=''))
parser$add_argument('--samp', type="integer",
                    help="Which sample number this should be saved as. Will also be used as random seed")
parser$add_argument('--n_train', type="integer", default=300000,
                    help="Number of items in the training set")
parser$add_argument('--n_test', type="integer", default=100000,
                    help="Number of items in the test set")

args <- parser$parse_args()

# Uncomment below to test in RStudio
# args <- data.frame(sim='A', samp=1, n_train=3000, n_test=1000)

# Set location
library(here)

# Load simulation functions
source("causal_experiment_simulator.R")

# Set random seed
set.seed(args$samp)

# Size of train and test
n_train <- args$n_train
n_test <- args$n_test

if(args$sim=='A'){
  # Simulation A (recreating SI Simulation 1)
  # Unbalanced case with simple CATE
  # alpha = default of 0.1 ?
  sim <- simulate_causal_experiment(
              ntrain = n_train,
              ntest = n_test,
              feat_distribution = "normal",
              dim = 20,
              pscore='rct01',
              mu0='simA',
              tau='simA'
  )
} else if(args$sim=='B'){
  # Simulation B (recreating SI Simulation 2)
  # Balanced case with complex linear CATE
  # alpha = default of 0.1 ?
  sim <- simulate_causal_experiment(
    ntrain = n_train,
    ntest = n_test,
    feat_distribution = "normal",
    dim = 20,
    pscore='rct5',
    mu0='simB',
    tau='simB'
  )
} else if(args$sim=='C'){
  # Simulation C (recreating SI Simulation 3)
  # Balanced case with complex non-linear CATE
  # alpha = default of 0.1 ?
  sim <- simulate_causal_experiment(
              ntrain = n_train,
              ntest = n_test,
              feat_distribution = "normal",
              dim = 20,
              pscore='rct5',
              mu0='simC',
              tau='simC'
  )
} else if(args$sim=='D'){
  # Simulation D (recreating SI Simulation 6)
  # Measured confounding with no TE
  sim <- simulate_causal_experiment(
              ntrain = n_train,
              ntest = n_test,
              feat_distribution = "unif",
              dim = 20,
              pscore='osSparse1Beta',
              mu0='simD',
              tau='no'
  )
} else if(args$sim=='E'){
  # Simulation E
  # Measured confounding with TE
  # NEED TO CHOOSE WHICH CATE:
  #     simple / complex linear / complex non-linear
  sim <- simulate_causal_experiment(
    ntrain = n_train,
    ntest = n_test,
    feat_distribution = "unif",
    dim = 20,
    pscore='osSparse1Beta',
    mu0='simD',
    tau='simA' # choosing simple CATE for now
  )
} else if(args$sim=='F'){
  # Simulation F
  # Unmeasured confounding with TE
  # tau <-- Beta --> pscore
  # tau = Beta added onto simple CATE 
  #       (adding confounding to tau in simE)

  sim <- simulate_causal_experiment(
    ntrain = n_train,
    ntest = n_test,
    feat_distribution = "unif",
    dim = 20,
    pscore='simF',
    mu0='simD',
    tau='simF'
  )
} else {
  stop(paste("Invalid simulation name. --sim must be one of: (",
             paste(working_sims, collapse=', '), ")", sep=''))
}

# Save train and test to dataframes
sim_train <-cbind(sim$feat_tr,
                   treatment = sim$W_tr,
                   Y = sim$Yobs_tr,
                   pscore = sim$Pscore_tr,
                   tau = sim$tau_tr)

sim_test <-cbind(sim$feat_te,
                  treatment = sim$W_te,
                  Y = sim$Yobs_te,
                  pscore = sim$Pscore_te,
                  tau = sim$tau_te)

# save train and test dataframes to parquet
file_train = here('data', paste('sim', args$sim, sep=''),
                  paste('samp', args$samp, '_train.parquet', sep='')
                  )
write_parquet(sim_train, file_train)

file_test = here('data', paste('sim', args$sim, sep=''),
            paste('samp', args$samp, '_test.parquet', sep='')
            )
write_parquet(sim_test, file_test)
