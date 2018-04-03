import torch
from torch.autograd import Variable
import numpy as np
from model import Model
from collector import Collector
from updater import Updater

n_envs = 100
n_tsteps = 50
n_bandits = 2
emb_size = 100
total_steps = 100000000
lr = 1e-4
gamma = 0.99
lambda_ = 0.96
val_coef = .1
entr_coef = .01
max_norm = .5
epochs_per_update = 1
norm_advs = False

# Make Model
net = Model(n_bandits, emb_size, batch_size=n_envs)
collector = Collector(net, n_envs, n_bandits)
updater = Updater(net,lr, gamma=gamma, lambda_=lambda_, val_coef=val_coef, entr_coef=entr_coef, max_norm=max_norm, norm_advs=norm_advs)

T = 0
avg_rew = 0
epoch = 0
while T < total_steps:    
    # Collect Rollout
    epoch += 1
    print("\nStep:", T, " – Epoch", epoch)
    data = collector.rollout(n_tsteps)
    T += n_tsteps*n_envs
    # Update Model
    updater.calc_loss(data)
    if epoch % epochs_per_update == 0:
        avg_rew = np.mean(data['rewards'])
        avg_val = np.mean(data['values'])
        avg_action = np.mean(data['sparse_actions'])
        print("Avg Rew:", avg_rew, " – Avg Act:", avg_action, " – Avg Val:",avg_val)
        print("Updating Model")
        updater.update_model(calc_grad=True)

        # Track Stats
        updater.print_stats()

print("All Done")
