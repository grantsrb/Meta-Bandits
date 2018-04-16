import torch
import torch.nn as nn
import numpy as np
from rnn.layer_norm import LayerNorm

"""
The Residual RNN (RRNN) is an RNN architecture introduced by Mujika in this paper: https://arxiv.org/abs/1611.01457

The structure is inspired by Residual Neural Network architectures.

Each step receives a previous state vector h_prev and a new value x. The RRNN then follows the equations:

r = f( concatenate( LN(h_prev), x) )
h = h_prev + r

Where LN is a Layer Normalization operation as detailed here: https://arxiv.org/abs/1607.06450
and f can be any differentiable function.

For this implementation we assume f to be a perceptron with a hyperbolic tangent activation function:

r = tanh( LN(h_prev).mm(W_h) + x.mm(W_x) + bias )
h = h_prev + r

Where W_h and W_x are trainable weight matrices. Note that concatenate(a, b).mm(W) == a.mm(W_a) + b.mm(W_b)
"""

class ResRNN(nn.Module):

    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(ResRNN, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 1
        self.layer_norm = layer_norm

        self.LN = LayerNorm((1,state_size))

        # h weight matrix initialization
        means = torch.zeros(state_size,state_size)
        rand_matrix = torch.normal(means, std=1/float(np.sqrt(state_size)))
        self.W_h = nn.Parameter(torch.FloatTensor(rand_matrix), requires_grad=True)

        # x weight matrix initialization
        means = torch.zeros(x_size,state_size)
        rand_matrix = torch.normal(means, std=1/float(np.sqrt(x_size)))
        self.W_x = nn.Parameter(torch.FloatTensor(rand_matrix), requires_grad=True)

        self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(1,state_size)), requires_grad=True)

        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):

        if self.layer_norm: h = self.LN(h_prev)
        else: h = h_prev
        r = self.tanh(x.mm(self.W_x) + h.mm(self.W_h) + self.bias)
        h = h_prev + r
        return (h,)
