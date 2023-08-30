import torch
import torch.nn as nn
import copy
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layer):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layer= n_layer
        self.rnn = nn.GRU(input_dim, hid_dim, n_layer)
    def forward(self, src, hidden):
        outputs, hidden = self.rnn(src, hidden)
        outputs = torch.add(outputs, src)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layer):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.rnn = nn.GRU(output_dim, hid_dim, n_layer)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        #input = input.veiw(1, -1)
        output, hidden = self.rnn(input, hidden)
        prediction = torch.add(input, self.fc_out(output))
        return prediction, hidden
class Seq2Seq(nn.Module):
    def __init__(self, goal_encoder, goal_decoder, kick_encoder, device):
        super().__init__()
        self.goal_encoder = goal_encoder
        self.goal_decoder = goal_decoder
        self.kick_encoder = kick_encoder
        self.w1 = torch.randn(1)
        self.w2 = torch.randn(1)
        self.device = device
    def forward(self, input_goal , input_kick , output):
        outputs = torch.zeros(output.shape[0], output.shape[1] , self.decoder.output_dim).to(self.device)
        encoder_stat_kick = torch.zeros()
        encoder_stat_goal = torch.zeros()
        for i in range(input.shape[0] - 1):
            encoder_out_goal, encoder_stat_goal = self.goal_encoder(input_goal[i], encoder_stat_goal)
            encoder_out_kick, encoder_stat_kick = self.kick_encoder(input_kick[i], encoder_stat_kick)
            temp = copy.deepcopy(encoder_stat_kick)
            encoder_stat_kick = torch.add(self.w1 * encoder_stat_kick, (1 - self.w1) * encoder_stat_goal)
            encoder_stat_goal = torch.add(self.w2 * temp, (1 - self.w2) * encoder_stat_goal)

        decoder_hidden = encoder_stat_goal.to(self.device)
        decoder_input = torch.tensor(input[-1], device=self.device)
        for t in range(output.shape[0]):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output
        return outputs