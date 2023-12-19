import torch
import torch.nn as nn
import numpy as np
import New_Seq2Seq_Model as NS
import sys
import animate as am
rnn_size = 1024
human_size = 51
file_path = sys.argv[1]
if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
goal_dataset = torch.from_numpy(np.load('./run/' + file_path, allow_pickle=True)['reconstruction']).reshape(2, -1, 51).to(dtype=torch.float32)
goal_dataset = goal_dataset.resize(-1, 51)
goal_encode = NS.Encoder(human_size, rnn_size, 1)
kick_encode = NS.Encoder(human_size, rnn_size, 1)
decode = NS.Decoder(human_size, rnn_size, 1)
model = NS.Seq2Seq(goal_encode, decode, kick_encode, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss().to(device)

checkpoint = torch.load('./model/checkpoint.pt')
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

z = torch.zeros(12,51)
out = model(goal_dataset, z)

am.draw_animation(out,  goal_dataset[-1, :])



