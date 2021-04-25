# Generate data for specific simulations
# Simulations A, B, C, and D recreate synthetic datasets from Kunzel et al

print(file.path(getwd(), "configurations", "netid.txt"))
netid = read.delim(file.path(getwd(), "configurations", "netid.txt"), 
                   header=FALSE, sep="\t"))
print(netid)
install_path = paste("/home/", netid, "/R/4.0.4", sep='')
print(install_path)

#install.packages("arrow", lib=install_path)
library("arrow", lib=install_path, warn.conflicts = FALSE)

# Valid simulations ready to be run
working_sims <- c('A', 'B', 'C', 'D', 'E', 'F')

# Get arguments from command line
#install.packages("argparse", lib=install_path)
library("argparse", lib=install_path)
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
# parser$add_argument('--netid', type="character",
#                     help="NYU NetID (used for saving output)")

args <- parser$parse_args()

# Uncomment below to test in RStudio
# args <- data.frame(sim='A', samp=1, n_train=3000, n_test=1000)

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
    pscore='osSparse1Beta',
    mu0='simD',
    tau='simA' # choosing simple CATE for now
  )
  
  # Get beta
  beta_tr <- (4*sim$Pscore_tr)-1
  beta_te <- (4*sim$Pscore_te)-1
  
  # Augment tau with beta
  sim$tau_tr <- sim$tau_tr + beta_tr
  sim$tau_te <- sim$tau_te + beta_te
  
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

# Path of output directory
output_dir = paste('/scratch/',netid,'/metalearners_data/sim',args$sim, sep='')

# Make output directory if necessary
mkdir(output_dir)

# Save training set
train_filepath = paste(output_dir,'/samp', args$samp, '_train.parquet', sep='')
write_parquet(sim_train, train_filepath)

# Save test set
test_filepath = paste(output_dir,'/samp', args$samp, '_test.parquet', sep='')
write_parquet(sim_test, test_filepath)
