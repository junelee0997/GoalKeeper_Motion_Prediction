import torch
import torch.nn as nn
import copy
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layer):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layer= n_layer
        self.rnn = nn.GRUCell(input_dim, hid_dim, n_layer)
    def forward(self, src, hidden):
        hidden = self.rnn(src, hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layer):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.rnn = nn.GRUCell(output_dim, hid_dim, n_layer)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        hidden = self.rnn(input, hidden)
        prediction = torch.add(input, self.fc_out(hidden))
        return prediction, hidden
class Seq2Seq(nn.Module):
    def __init__(self, goal_encoder, goal_decoder, kick_encoder, device):
        super().__init__()
        self.goal_encoder = goal_encoder
        self.goal_decoder = goal_decoder
        self.kick_encoder = kick_encoder
        self.hid_out = nn.Linear(2048, 1024)
        self.hid_out2 = nn.Linear(2048, 1024)
        self.device = device
    def forward(self, input , output):
        outputs = torch.zeros(output.shape[0], output.shape[1], self.goal_decoder.output_dim, dtype=torch.float32).to(self.device)
        encoder_stat_kick = torch.zeros(input.shape[0], self.kick_encoder.hid_dim, dtype=torch.float32).to(self.device)
        encoder_stat_goal = torch.zeros(input.shape[0], self.goal_encoder.hid_dim, dtype=torch.float32).to(self.device)
        encode_size = input.shape[1] // 2
        for i in range(encode_size):
            encoder_stat_goal = self.goal_encoder(input[:, i, :], encoder_stat_goal)
            encoder_stat_kick = self.kick_encoder(input[:, i + encode_size + 1, :], encoder_stat_kick)
            temp = torch.concat((encoder_stat_goal, encoder_stat_kick), dim=1)
            encoder_stat_kick = self.hid_out(temp)
            encoder_stat_goal = self.hid_out2(temp)

        decoder_hidden = encoder_stat_goal.to(self.device)
        decoder_input = input[:, encode_size-1, :].to(self.device)
        for t in range(output.shape[1]):
            decoder_output, decoder_hidden = self.goal_decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = decoder_output
            decoder_input = decoder_output
        return outputs