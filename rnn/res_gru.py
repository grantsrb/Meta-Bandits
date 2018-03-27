import torch
import torch.nn as nn
import numpy as np
from rnn.gru import GRU 

"""
This GRU unit uses a residual addition to the h term via the following formulae:

z = sigmoid(W_x_z.mm(x) + W_h_z.mm(old_h) + b_z)
r = sigmoid(W_x_r.mm(x) + W_h_r.mm(old_h) + b_r
h = z*old_h + (1-z)*tanh(W_x_h.mm(x) + W_h_h.mm(r*old_h) + b_h) + old_h

Where x is the new, incoming data and old_h is the h at the previous time step.
Each of the W_x_ and W_h_ terms are weight matrices and the b_ terms are biases.
In this implementation, all of the W_x_ terms are combined into a single variable. Same
with the W_h_ and b_ terms.
"""

class ResGRU(nn.Module):
    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(ResGRU, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 1

        self.gru = GRU(x_size, state_size, layer_norm=layer_norm)


    def forward(self, x, old_h):
        """
        old_h - running state of GRU. FloatTensor Variable with shape (batch_size, state_size)
        x - New data coming into GRU. FloatTensor Variable with shape (batch_size, state_size)
        """
        h = self.gru(x,old_h)
        return (h[0]+old_h,)

    def check_grads(self):
        for p in list(self.parameters()):
            if torch.sum(p.grad.data != p.grad.data):
                print("NaN in grads")
