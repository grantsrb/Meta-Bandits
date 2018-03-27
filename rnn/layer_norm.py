import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, shape, epsilon=1e-4):
        super(LayerNorm, self).__init__()
        """
        shape - shape of the features for a single piece of data ignoring batch_size 
                (shape_of_data,) or (1,shape_of_data)
        epsilon - small constant to prevent division by 0
        """
        self.g = nn.Parameter(torch.ones(*shape))
        self.b = nn.Parameter(torch.zeros(*shape))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean()
        std = x.std()
        return self.g*(x - mean)/(std + self.epsilon) + self.b
        
