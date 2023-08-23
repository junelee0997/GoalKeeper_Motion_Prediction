import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layer):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layer= n_layer
        self.rnn = nn.GRU(input_dim, hid_dim, n_layer)

    def forward(self, src):
        outputs, hidden = self.rnn(src)
        outputs = torch.add(outputs, src)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layer):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.rnn = nn.GRU(input_dim, hid_dim, n_layer)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        #input = input.veiw(1, -1)
        output, hidden = self.rnn(input, hidden)
        prediction = torch.add(input, self.fc_out(output))
        return prediction, hidden
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, input, output):
        outputs = torch.zeros(output.shape[0], output.shape[1] , self.decoder.output_dim).to(self.device)
        for i in range(input.shape[0] - 1):
            encoder_out, encoder_stat = self.encoder(input[i])
        decoder_hidden = encoder_stat.to(self.device)
        decoder_input = torch.tensor(input[-1], device=self.device)
        for t in range(output.shape[0]):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output
        return outputs