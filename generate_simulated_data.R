# Generate data for specific simulations
# Simulations A, B, C, and D recreate synthetic datasets from Kunzel et al


train_size <- 300000


# Simulation A (recreating SI Simulation 1)
simulated_experimenta <- simulate_causal_experiment(
  ntrain = train_size,
  ntest = 100000,
  feat_distribution = "normal",
  dim = 20,
  pscore='rct01',
  mu0='simA',
  tau='simA'
)

# Simulation B (recreating SI Simulation 2)
# simulated_experiment2 <- simulate_causal_experiment(
#   ntrain = train_size,
#   ntest = 100000,
#   feat_distribution = "normal",
#   dim = 20,
#   pscore='rct5',
#   mu0='simB',
#   tau= ???
# )

# Simulation C (recreating SI Simulation 3)
simulated_experiment3 <- simulate_causal_experiment(
  ntrain = train_size,
  ntest = 100000,
  feat_distribution = "normal",
  dim = 20,
  pscore='rct5',
  mu0='simC',
  tau='simC'
)

# Simulation D (recreating SI Simulation 6)
simulated_experiment3 <- simulate_causal_experiment(
  ntrain = train_size,
  ntest = 100000,
  feat_distribution = "unif",
  dim = 20,
  pscore='osSparse1Beta',
  mu0='simD',
  tau='no'
)