#!/usr/bin/env python3

class configuration:
    def __init__(self, metalearner):
        self.metalearner = metalearner

class Tconfig(configuration):
    def __init__(self, mu_0, mu_1):
        self.metalearner = 'T'
        self.mu_0 = baseLearner(mu_0,'mu_0')
        self.mu_1 = baseLearner(mu_1,'mu_1')

    def set_all_hyperparams(self, hp_dict, sim):
        self.mu_0.set_hyperparams(hp_dict, sim)
        self.mu_1.set_hyperparams(hp_dict, sim)

class Sconfig(configuration):
    def __init__(self, mu):
        self.metalearner = 'S'
        self.mu = baseLearner(mu,'mu')

    def set_all_hyperparams(self, hp_dict, sim):
        self.mu.set_hyperparams(hp_dict, sim)


class Xconfig(configuration):
    def __init__(self, mu_0, mu_1,
                 tau_0, tau_1, g):
        self.metalearner = 'X'
        self.mu_0 = baseLearner(mu_0, 'mu_0')
        self.mu_1 = baseLearner(mu_1,'mu_1')
        self.tau_0 = baseLearner(tau_0,'tau_0')
        self.tau_1 = baseLearner(tau_1,'tau_1')
        self.g = baseLearner(g,'g')

    def set_all_hyperparams(self, hp_dict, sim):
        self.mu_0.set_hyperparams(hp_dict, sim)
        self.mu_1.set_hyperparams(hp_dict, sim)
        self.tau_0.set_hyperparams(hp_dict, sim)
        self.tau_1.set_hyperparams(hp_dict, sim)
        self.g.set_hyperparams(hp_dict, sim)

class baseLearner:
    def __init__(self, name, est):
        self.algo=name
        self.estimand=est

    # Needs to be updated for RandomForestClassifier params (for g)
    def set_hyperparams(self, hp_dict, sim):
        if self.algo=='rf':
            self.hyperparams = hp_dict[self.estimand]['sim'+sim]
        else:
            self.hyperparams = {}