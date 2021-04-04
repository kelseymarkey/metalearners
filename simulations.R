#simulation 1
simulated_experiment1 <- simulate_causal_experiment(
  ntrain = 300000,
  ntest = 100000,
  feat_distribution = "normal",
  dim = 20,
  pscore='rct01',
  mu0='sim1',
  tau='sim1'
)

feature_train <- simulated_experiment1$feat_tr
w_train <- simulated_experiment1$W_tr
yobs_train <- simulated_experiment1$Yobs_tr
simulated_experiment1_df <- cbind(feature_train, w_train, yobs_train)

#simulation 3
simulated_experiment3 <- simulate_causal_experiment(
  ntrain = 300000,
  ntest = 100000,
  feat_distribution = "normal",
  dim = 20,
  pscore='rct5',
  mu0='sim3',
  tau='sim3'
)

feature_train <- simulated_experiment3$feat_tr
w_train <- simulated_experiment3$W_tr
yobs_train <- simulated_experiment3$Yobs_tr
simulated_experiment3_df <- cbind(feature_train, w_train, yobs_train)


write.csv(simulated_experiment1_df,"C:\\Users\\laurendarinzo\\Desktop\\metalearners\\simulation1.csv")
write.csv(simulated_experiment3_df,"C:\\Users\\laurendarinzo\\Desktop\\metalearners\\simulation3.csv")

write.csv(simulated_experiment1_df, 'simulation1.csv')
write.csv(simulated_experiment3_df, 'simulation3.csv')
          