import torch
from torch.autograd import Variable
import numpy as np
from model import Model
from collector import Collector
from updater import Updater
from logger import Logger

def cuda_if(tobj):
    if torch.cuda.is_available():
        tobj = tobj.cuda()
    return tobj

save_file = 'default'
n_envs = 100
n_tsteps = 20
n_bandits = 2
emb_size = 100
total_steps = 100000000
lr = 1e-4
gamma = 0.96
lambda_ = 0.94
val_coef = .008
entr_coef = .0009
max_norm = .5
epochs_per_update = 2
bandit_prob = .9
norm_advs = True
bootstrap = True
resume = False
evaluate_mode = True

if evaluate_mode:
    n_envs = 1
    total_steps = n_tsteps*10

# Make Model
net = Model(n_bandits, emb_size, batch_size=n_envs)
if resume or evaluate_mode:
    net.load_state_dict(torch.load(save_file+"_net.p"))
net = cuda_if(net)
collector = Collector(net, n_envs, n_bandits, bandit_prob, bootstrap=bootstrap)
updater = Updater(net, lr, gamma=gamma, lambda_=lambda_, val_coef=val_coef, entr_coef=entr_coef, max_norm=max_norm, norm_advs=norm_advs, bootstrap=bootstrap)
if resume:
    updater.optim.load_state_dict(torch.load(save_file+"_optim.p"))

# Make logger
logger = Logger()

T = 0
avg_rew = 0.5
epoch = 0
trial = 0

while T < total_steps:    
    epoch += 1
    # Collect Rollout
    data = collector.rollout(n_tsteps)
    avg_rew = .99*avg_rew + .01*np.mean(data['rewards'])
    T += n_tsteps*n_envs
    # Update Model
    if evaluate_mode:
        logger.data["Trial "+str(epoch)] = np.asarray(data['rewards'])
    else:
        updater.calc_loss(data)
        if epoch % epochs_per_update == 0:
            print("\nStep:", T, " – Epoch", epoch)
            avg_val = np.mean(data['values'])
            avg_action = np.mean(data['sparse_actions'])
            print("Avg Rew:", avg_rew, " – Avg Act:", avg_action, " – Avg Val:",avg_val)
            updater.update_model(calc_grad=True)

            # Track Stats
            updater.print_stats()
            updater.info["Avg Reward"] = avg_rew
            updater.info["Avg Action"] = avg_action
            updater.info["Avg Value Pred"] = avg_val
            logger.append(updater.info, x_val=T)
            updater.save_model(save_file)

if evaluate_mode:
    cat = np.vstack([logger.data[key] for key in logger.data.keys() if 'x_vals' not in key])
    logger.data['Avg Reward Across Trials'] = cat.mean(0)

logger.make_plots()
print("All Done")
