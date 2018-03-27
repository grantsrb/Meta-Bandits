import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Collector():

    def cpu_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cpu()
        return t_obj

    def cuda_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cuda()
        return t_obj

    def __init__(self, net, envs, n_tsteps):
        self.net = net # PyTorch Module
        self.envs = envs # Vector of bandit envs
        self.n_tsteps = n_tsteps
        self.pi_space = self.envs[0].n_bandits
        self.softmax = envs[0].softmax

    def rollout(self):
        data = {'actions':[], 'rewards':[], 'values':[]}
        self.net.reset_state(len(self.envs))
        self.net.train(mode=False)
        net_input = Variable(self.cuda_if(torch.zeros(len(self.envs),self.pi_space+1)))
        net_inputs = self.cuda_if(torch.zeros(self.n_tsteps,len(self.envs),self.pi_space+1))
        net_inputs[0] = net_input.data
        for i in range(1, self.n_tsteps+1):
            outputs,vals = self.net.forward(net_input)
            pis = self.softmax(self.cpu_if(outputs.data).numpy())
            actions = self.get_actions(pis)
            rewards = self.get_rewards(actions)
            data['actions'].append(actions)
            data['rewards'].append(rewards)
            data['values'].append(self.cpu_if(vals.data).numpy())
            net_input = self.get_net_input(actions, rewards)
            net_inputs[i] = net_input
        
        outputs,vals = self.net.forward(net_input)
        data['values'].append(self.cpu_if(vals.data).numpy())
        for key in data.keys():
            data[key] = np.asarray(data[key], dtype=np.float32)
        data['net_inputs'] = net_inputs
        return data

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
        return actions.astype(np.float32)
    
    def get_rewards(self, actions):
        """
        actions - ndarray of one_hot actions. shape (batch_size, n_bandits)

        returns:
            rewards - ndarray of collected rewards. shape = (batch_size, 1)
        """
        rewards = []
        for action,env in zip(actions,envs):
            rewards.append(env.pull_lever(action))
        return np.asarray(rewards, dtype=np.float32)[None]

    def get_net_input(self, actions, rewards):
        """
        actions - ndarray of one_hot actions. shape (batch_size, n_bandits)
        rewards - ndarray of collected rewards with shape (batch_size, 1)

        returns:
            Variable torch FloatTensor of shape (batch_size, n_bandits+1)
        """
        cats = np.concatenate([actions, rewards], axis=-1)
        return Variable(self.cuda_if(torch.FloatTensor(cats)))


