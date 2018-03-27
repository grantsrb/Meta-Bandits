import torch
from torch.autograd import Variable
import numpy as np
import model
import bandits


"""
PseudoCode:
    Create a number of bandits with random weights 
    Collect rollout of actions
    Update Model
    Repeat
"""

n_envs = 100
n_tsteps = 100
n_bandits = 2
emb_size = 100
total_steps = 10000
lr = 1e-3

# Make bandits
envs = []
for i in range(n_envs):
    envs.append(bandits.Bandits(n_bandits))

# Make Model
net = model.Model(n_bandits, batch_size=n_envs)
collector = Collector(net, envs, n_tsteps)
updater = Updater(net,lr)

T = 0
avg_rew = 0
while T < total_steps:    
    # Collect Rollout
    data = collector.rollout()
    T += len(data[0])
    avg_rew = np.mean(data['reward'])

    # Update Model
    updater.update_model(data)

    # Track Stats
    updater.print_stats()


    
