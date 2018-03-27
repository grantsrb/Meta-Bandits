import torch
import torch.nn as nn
import rnn.GRU as GRU

class Model(nn.Module):

    def cuda_if(self, t_obj):
        if torch.cuda.is_available():
            t_obj = t_obj.cuda()
        return t_obj

    def __init__(self, n_bandits, state_size, batch_size=1):
        super(Model, self).__init__()
        self.n_bandits = n_bandits
        self.state_size = state_size
        self.batch_size = batch_size

        self.bandits = nn.Parameter(torch.ones(n_bandits).float())
        means = torch.zeros(state_size+n_bandits+1, n_bandits)
        self.output = nn.Parameter(torch.normal(means=means, std=1/torch.sqrt(state_size)))
        self.state = self.reset_state(batch_size)
        self.gru = GRU(x_size=n_bandits, state_size=state_size)

    def forward(self, last_data):
        """
        last_data - Variable torch FloatTensor of the last action (one hot encoded) 
                    concatenated with the last reward. shape = (batch_size, n_bandits+1)
        """
        new_state = self.gru(last_data, Variable(self.state))
        aprobs = self.output(new_state)
        self.state = new_state.data
        return aprobs

    def reset_state(self, batch_size=None):
        """
        Clears the state and makes a new one of shape (length, self.state_size)
        """
        if batch_size is not None: self.batch_size = batch_size
        self.state = torch.zeros(self.batch_size, self.state_size)
        return self.cuda_if(self.state)

