import torch
from torch.autograd import Variable
import numpy as np
from model import Model
import bandits
from collector import Collector
from updater import Updater



n_envs = 100
n_tsteps = 100
n_bandits = 2
emb_size = 100
total_steps = 1000000
lr = 1e-3
gamma = 0.99
lambda_ = 0.96
val_coef = .5
entr_coef = .01
max_norm = .5

# Make bandits
envs = []
for i in range(n_envs):
    envs.append(bandits.Bandits(n_bandits))

# Make Model
net = Model(n_bandits, emb_size, batch_size=n_envs)
collector = Collector(net, envs, n_tsteps)
updater = Updater(net,lr, gamma=gamma, lambda_=lambda_, val_coef=val_coef, entr_coef=entr_coef, max_norm=max_norm)

T = 0
avg_rew = 0
while T < total_steps:    
    # Collect Rollout
    print("\nStep:", T)
    data = collector.rollout()
    T += n_tsteps*n_envs
    avg_rew = np.mean(data['rewards'])
    print("Avg Rew:", avg_rew)

    # Update Model
    updater.calc_loss(data)
    updater.update_model(calc_grad=True)

    # Track Stats
    updater.print_stats()

print("All Done")
