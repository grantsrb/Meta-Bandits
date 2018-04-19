import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Updater():
    
    def cuda_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cuda()
        return t_obj

    def cpu_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cpu()
        return t_obj

    def __init__(self, net, lr, gamma=.995, lambda_=.95, val_coef=.5, entr_coef=.01, max_norm=.5, norm_advs=False, bootstrap=True):
        self.net = net
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.val_coef = val_coef
        self.entr_coef = entr_coef
        self.max_norm = max_norm
        self.norm_advs = norm_advs
        self.bootstrap = bootstrap
        self.global_loss = 0
        self.pi_loss = 0
        self.val_loss = 0
        self.entr = 0
        self.info = {"Global":self.global_loss,"Pi":self.pi_loss,"Val":self.val_loss,"Entr":self.entr}

    def calc_loss(self, data):
        """
        data - dict with keys:
                "rewards" - ndarray of rewards collected in the rollout. 
                            shape (n_tsteps, n_envs)
                "values" - ndarray of value predictions collected in the rollout. 
                            shape (n_tsteps+1, n_envs) 
                "actions" - ndarray of one_hot encoded actions collected 
                            in the rollout. shape (n_tsteps, n_envs, n_bandits)
                "net_inputs" - torch FloatTensor of the net inputs at each step 
                            in the rollout. shape (n_tsteps, n_envs, n_bandits)
        """
        
        rewards = data['rewards']
        actions = data['actions']
        values = data['values']
        net_inputs = data['net_inputs']

        # Make Advantages
        advantages = self.gae(rewards, values, self.gamma, self.lambda_)
        #returns = self.discount(rewards, self.gamma)
        returns = advantages + values[:-1]
        if self.norm_advs:
            advantages = (advantages - np.mean(advantages))
            advantages = advantages/(np.max(advantages)+1e-6)
        advantages = Variable(self.cuda_if(torch.FloatTensor(advantages)))

        # Make Action Probs and Vals
        net_inputs = Variable(self.cuda_if(net_inputs))
        raw_pis = Variable(self.cuda_if(torch.zeros(actions.shape))) # Storage variable for efficiency
        vals = Variable(self.cuda_if(torch.zeros(rewards.shape))) # Storage variable for efficiency
        self.net.reset_state(actions.shape[1])
        self.net.train(mode=True)
        self.net.req_grads(True)
        for i in range(len(rewards)):
            inputs = net_inputs[i]
            raw_prob, val = self.net.forward(inputs)
            raw_pis[i] = raw_prob
            vals[i] = val

        # Policy Loss
        actions = Variable(self.cuda_if(torch.FloatTensor(actions)))
        log_probs = F.log_softmax(raw_pis, dim=-1)
        log_pis = torch.sum(log_probs*actions, dim=-1)
        pi_loss = log_pis*advantages.squeeze()
        pi_loss = -(pi_loss).mean()

        # Value Loss
        targets = Variable(self.cuda_if(torch.FloatTensor(returns)))
        val_loss = self.val_coef*F.mse_loss(vals.squeeze(), targets.squeeze())
        #val_loss = Variable(torch.zeros(1))

        # Entropy
        probs = F.softmax(raw_pis, dim=-1)
        entropy = torch.sum(probs*log_probs, dim=-1).mean()
        entropy = -self.entr_coef*entropy

        self.global_loss += pi_loss + val_loss - entropy
        #self.global_loss += pi_loss - entropy
        self.pi_loss += pi_loss.data[0]
        self.val_loss += val_loss.data[0]
        self.entr += entropy.data[0]

    def calc_gradients(self):
        try:
            self.global_loss.backward()
            self.info = {"Global":self.global_loss.data[0],"Pi":self.pi_loss,"Val":self.val_loss,"Entr":self.entr}
            self.global_loss = 0
            self.pi_loss = 0
            self.val_loss = 0
            self.entr = 0
        except RuntimeError:
            print("Attempted backwards pass with no graph")

    def update_model(self, calc_grad=True):
        """
        Calculates gradient and performs step of gradient descent.
        """
        if calc_grad:
            self.calc_gradients()
        self.norm = nn.utils.clip_grad_norm(self.net.parameters(), self.max_norm)
        self.optim.step()
        self.optim.zero_grad()

    def print_stats(self):
        print("\n".join([key+": "+str(round(val,8)) for key,val in self.info.items()]))

    def gae(self, rewards, values, gamma, lambda_):
        """
        rewards - ndarray of rewards collected in the rollout. Rewards are assumed to 
                    have included the bootstrapped value. shape (n_tsteps, n_envs)
        values - ndarray of value predictions collected in the rollout. 
                   shape (n_tsteps+1, n_envs)
        gamma - discount factor between 0 and 1
        lambda_ - gae moving avg factor between 0 and 1
        """
        advs = rewards + gamma*values[1:] - values[:-1]
        return self.discount(advs, gamma*lambda_)
         
    def discount(self, rewards, disc_factor):
        """
        rewards - ndarray of rewards to be discounted, shape = (n_tsteps, n_envs)
        disc_factor - decay constant float between 0 and 1

        returns:
            disc_rews - discounted rewards of shape (n_tsteps, n_envs)
        """
        disc_rews = np.zeros_like(rewards)
        running_sum = np.zeros(rewards.shape[1:])
        for i in reversed(range(len(rewards))):
            running_sum = rewards[i] + disc_factor*running_sum
            disc_rews[i] = running_sum.copy()
        return disc_rews
    
    def save_model(self, save_file):
        torch.save(self.net.state_dict(), save_file+"_net.p")
        torch.save(self.optim.state_dict(), save_file+"_optim.p")

