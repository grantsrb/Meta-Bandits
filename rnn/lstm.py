import torch
import torch.nn as nn
import numpy as np
from rnn.layer_norm import LayerNorm

"""
LSTM units follow the formulae:

i = sigmoid(x.mm(W_x_i) + old_h.mm(W_h_i) + b_i)
f = sigmoid(x.mm(W_x_f) + old_h.mm(W_h_f) + b_f)
c = f*old_c + i*tanh(x.mm(W_x_c) + old_h.mm(W_h_c) + b_c)
o = sigmoid(x.mm(W_x_o) + old_h.mm(W_h_o) + b_o)
h = o*tanh(c)

Where x is the new, incoming data old_h is the h at the previous time step, and
old_c is the c at the previous time step.
Each of the W_ and W_h_ terms are weight matrices and the b_ terms are biases specific
to the quantity being calculated.

"""

class LSTM(nn.Module):
    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(LSTM, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 2

        # Internal LSTM Entry Parameters
        means = torch.zeros(4, x_size, state_size)
        self.W_x = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM State Parameters
        means = torch.zeros(4, state_size, state_size)
        self.W_h = nn.Parameter(torch.normal(means, std=1/float(np.sqrt(state_size))), requires_grad=True)

        # Internal LSTM Bias Parameters
        self.b = nn.Parameter(torch.zeros(4,1,state_size), requires_grad=True)

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Layer Normalization
        self.layer_norm = layer_norm
        if layer_norm:
            self.LN = LayerNorm((1,state_size))

    def forward(self, x, old_h, old_c):
        """
        x - New data coming into LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_h - short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)

        returns:
            h - new short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c - new long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        """

        if self.layer_norm:
            old_h = self.LN(old_h)
            old_c = self.LN(old_c)
        i = self.sigmoid(x.mm(self.W_x[0]) + old_h.mm(self.W_h[0]) + self.b[0])
        f = self.sigmoid(x.mm(self.W_x[1]) + old_h.mm(self.W_h[1]) + self.b[1])
        c = f*old_c + i*self.tanh(x.mm(self.W_x[2]) + old_h.mm(self.W_h[2]) + self.b[2])
        o = self.sigmoid(x.mm(self.W_x[3]) + old_h.mm(self.W_h[3]) + self.b[3])
        h = o*self.tanh(c)

        return h, c
