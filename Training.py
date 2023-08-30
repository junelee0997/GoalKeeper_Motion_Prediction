import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import New_Seq2Seq_Model as NS
import data_util as du
import transform as tf

rnn_size = 1024
human_size = 42
max_norm = 5
batch_size = 16
data_count = 1000
iterations = 100000
epochs = batch_size * iterations // data_count

goal_dataset, kick_dataset = du.data_splice()
goal_dataset = tf.make_joint_vector(goal_dataset)
kick_dataset = tf.make_joint_vector(kick_dataset)

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
goal_encode = NS.Encoder(human_size, rnn_size, 1)
kick_encode = NS.Encoder(human_size, rnn_size, 1)
decode = NS.Decoder(human_size, rnn_size, 1)
model = NS.Seq2Seq(goal_encode,kick_encode, decode, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = torch.nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000,gamma=0.95)



'''train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

for epoch in range():

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
print(encode)'''