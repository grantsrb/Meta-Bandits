import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import bandit

class Collector():

    def cpu_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cpu()
        return t_obj

    def cuda_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cuda()
        return t_obj

    def __init__(self, net, n_envs, n_bandits):
        self.net = net # PyTorch Module
        self.pi_space = n_bandits
        self.n_envs = n_envs
        self.softmax = bandit.Bandit().softmax

    def rollout(self, n_tsteps):
        self.envs = self.new_envs() # Vector of bandit envs
        data = {'actions':[], 'sparse_actions':[], 'rewards':[], 'values':[]}
        self.net.reset_state(len(self.envs))
        self.net.train(mode=False)
        net_input = Variable(self.cuda_if(torch.zeros(len(self.envs),self.pi_space+1)))
        net_inputs = self.cuda_if(torch.zeros(n_tsteps,len(self.envs),self.pi_space+1))
        for i in range(0, n_tsteps):
            net_inputs[i] = net_input.data
            outputs,vals = self.net.forward(net_input)
            pis = self.softmax(self.cpu_if(outputs.data).numpy())
            actions = self.get_actions(pis)
            rewards = self.get_rewards(actions)
            data['sparse_actions'].append(np.argmax(actions, axis=-1))
            data['actions'].append(actions)
            data['rewards'].append(rewards)
            data['values'].append(self.cpu_if(vals.data.squeeze()).numpy())
            net_input = self.get_net_input(actions, rewards)
        
        outputs, vals = self.net.forward(net_input)
        data['values'].append(self.cpu_if(vals.data.squeeze()).numpy())
        for key in data.keys():
            data[key] = np.asarray(data[key], dtype=np.float32)
        data['rewards'] = data['rewards'].squeeze()
        data['net_inputs'] = net_inputs
        return data

    def new_envs(self):
        """
        Makes a new list of bandit environments.
        """
        envs = []
        for i in range(self.n_envs):
            rand = np.random.random()
            probs = [.1,.9] if rand <= 0.5 else [.9,.1]
            envs.append(bandit.Bandit(probs=probs))
        return envs

    def get_actions(self, pis):
        """
        pis - ndarray of action probabilities shape (batch_size, n_bandits)

        returns:
            actions - ndarray of one hot selected actions, shape = (batch_size, n_bandits)
        """
        if len(pis.shape) < 2: pis = pis[None]
        cumsums = np.zeros((pis.shape[0],))
        randoms = np.random.random((pis.shape[0],))
        actions = np.zeros(pis.shape)
        for i in range(pis.shape[1]):
            cumsums += pis[:,i]
            actions[:,i] = (randoms < cumsums)
            randoms += actions[:,i]
        actions = actions.astype(np.float32)
        return actions
    
    def get_rewards(self, actions):
        """
        actions - ndarray of one_hot actions. shape (batch_size, n_bandits)

        returns:
            rewards - ndarray of collected rewards. shape = (batch_size, 1)
        """
        rewards = []
        for action,env in zip(actions,self.envs):
            rewards.append(env.pull_lever(action))
        return np.asarray(rewards, dtype=np.float32)[..., None]

    def get_net_input(self, actions, rewards):
        """
        actions - ndarray of one_hot actions. shape (batch_size, n_bandits)
        rewards - ndarray of collected rewards with shape (batch_size, 1)

        returns:
            Variable torch FloatTensor of shape (batch_size, n_bandits+1)
        """
        cats = np.concatenate([actions, rewards], axis=-1)
        return Variable(self.cuda_if(torch.FloatTensor(cats)))


