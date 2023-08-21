import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.module):
    def __init__(self, input_dim, hid_dim, n_layer, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layer= n_layer
        self.rnn = nn.GRU(input_dim, hid_dim, n_layer, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        encoded = self.dropout(src)
        outputs, (hidden, cell) = self.rnn(encoded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layer, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.rnn = nn.GRU(output_dim, hid_dim, n_layer, n_layer, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        encoded = self.dropout(input)
        output, (hidden, cell) = self.rnn(encoded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim, \
            'Hidden dimensions of encoder decoder must be equal'
        assert encoder.n_layers == decoder.n_layers, \
            'Encoder and decoder must have equal number of layers'