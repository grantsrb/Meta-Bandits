import torch
import torch.nn as nn
import numpy as np
from rnn.nested_lstm import NestedLSTM
from rnn.layer_norm import LayerNorm

"""
This implementation includes a skip connection from the h state variable to the output h statevariable,
with a layer normalization on each hidden state variable.
This Residual Nested LSTM follows this formula:

old_h, old_c, old_c_hat = LayerNorm(old_h), LayerNorm(old_c), LayerNorm(old_c_hat)
i = sigmoid(x.mm(W_x_i) + old_h.mm(W_h_i) + b_i)
f = sigmoid(x.mm(W_x_f) + old_h.mm(W_h_f) + b_f)

old_h_hat = f*old_c
x_hat = i*tanh(x.mm(W_x_c) + old_h.mm(W_h_c) + b_c)

i_hat = sigmoid(x_hat.mm(W_x_i_hat) + old_h_hat.mm(W_h_i_hat) + b_i_hat)
f_hat = sigmoid(x_hat.mm(W_x_f_hat) + old_h_hat.mm(W_h_f_hat) + b_f_hat)
c_hat = f_hat*old_c_hat + i_hat*tanh(x_hat.mm(W_x_c_hat) + old_h_hat.mm(W_h_c_hat) + b_c_hat)
o_hat = sigmoid(x_hat.mm(W_x_o_hat) + old_h_hat.mm(W_h_o_hat) + b_o_hat)
h_hat = o_hat*tanh(c_hat)
c = h_hat
o = sigmoid(x.mm(W_x_o) + old_h.mm(W_h_o) + b_o)
h = o*tanh(c) + old_h

Where x is the new, incoming data old_h is the h at the previous time step, and
old_c is the c at the previous time step, and old_c_hat is the c_hat at the previous time step.
Each of the W_x_ and W_h_ terms are weight matrices and the b_ terms are biases specific
to the quantity being calculated.

"""

class ResNestedLSTM(nn.Module):
    def __init__(self, x_size=256, state_size=256, layer_norm=False):
        super(ResNestedLSTM, self).__init__()

        self.x_size = x_size
        self.state_size = state_size
        self.n_state_vars = 3

        # Nested LSTM
        self.nested_lstm = NestedLSTM(x_size, state_size)

        # Layer Norm
        self.layer_norm = layer_norm
        if layer_norm:
            self.LN_h = LayerNorm((1,state_size))
            self.LN_c = LayerNorm((1,state_size))
            self.LN_c_hat = LayerNorm((1,state_size))

        # Non Linear Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, old_h, old_c, old_c_hat):
        """
        x - New data coming into LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_h - short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
        old_c - long term memory of nested LSTM. FloatTensor Variable with shape (batch_size, state_size)

        returns:
            h - new short term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c - new long term memory of LSTM. FloatTensor Variable with shape (batch_size, state_size)
            c_hat - new long term memory of nested LSTM. FloatTensor Variable of shape (batch_size, state_size)
        """

        if self.layer_norm:
            old_h = self.LN_h(old_h)
            old_c = self.LN_c(old_c)
            old_c_hat = self.LN_c_hat(old_c_hat)

        h,c,c_hat = self.nested_lstm(x, old_h, old_c, old_c_hat)

        return h+old_h, c, c_hat
