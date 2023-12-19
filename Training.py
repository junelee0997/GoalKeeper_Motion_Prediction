import torch
import torch.nn as nn
import numpy as np
import New_Seq2Seq_Model as NS
import transform as tf
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import os
from os import path

rnn_size = 1024
human_size = 51
max_norm = 5
batch_size = 16
data_count = 37#1000
iterations = 100000
check_step = 1000
max_frame = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

firstdir = os.listdir('./data')[0]
goal_dataset = torch.from_numpy(np.load('./data/' + firstdir, allow_pickle=True)['reconstruction']).reshape(2, -1, 51)
goal_train = torch.from_numpy(np.load('./train/' + firstdir[:-4] + '_train.npz', allow_pickle=True)['reconstruction'][1]).reshape(-1, 51)[:max_frame]
goal_dataset = tf.make_joint_vector(goal_dataset)
goal_train = tf.make_joint_vector(goal_train)
seq_len = [goal_dataset.shape[0]]
goal_dataset = [goal_dataset]
for i in os.listdir('./data'):
    if i == firstdir:
        continue
    ngoal_dataset = (torch.from_numpy(np.load('./data/' + i, allow_pickle=True)['reconstruction']).to(dtype=torch.float32).reshape(2, -1, 51))
    ngoal_dataset = tf.make_joint_vector(ngoal_dataset)
    seq_len.append(ngoal_dataset.shape[0])
    ngoal_train = torch.from_numpy(np.load('./train/' + i[:-4] + '_train.npz', allow_pickle=True)['reconstruction'][1]).to(dtype=torch.float32).reshape(-1, 51)[:max_frame]
    ngoal_train = tf.make_joint_vector(ngoal_train)
    goal_train = tf.data_bind(goal_train, ngoal_train)
    goal_dataset.append(ngoal_dataset)

seq_len = torch.tensor(seq_len)
pack = torch.nn.utils.rnn.pad_sequence(goal_dataset, batch_first=True)

dataset = TensorDataset(pack, goal_train)
d_length = (int)(len(dataset) * 0.8)
ev_length = len(dataset) - d_length
dataset, eval = random_split(dataset, [d_length, ev_length])
dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
eval = DataLoader(dataset=eval, batch_size=batch_size, shuffle=False)

torch.manual_seed(1)

goal_encode = NS.Encoder(human_size, rnn_size, 1)
kick_encode = NS.Encoder(human_size, rnn_size, 1)
decode = NS.Decoder(human_size, rnn_size, 1)
model = NS.Seq2Seq(goal_encode, decode, kick_encode, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000 // data_count,gamma=0.95)

ep = 1
if path.exists('./model/checkpoint.pt'):
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ep = checkpoint["epoch"]

data_count = len(dataset)
epochs = batch_size * iterations // data_count

for epoch in range(ep, epochs + 1):
    model.train()
    for x, t in enumerate(dataset):
        y, z = t
        prediction = model(y, z)
        loss = criterion(prediction, z)
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, epochs, x + 1, len(dataset),
            loss.item()
        ))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        scheduler.step()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in eval:
            pred = model(x, y)
            loss = criterion(pred, y)
            test_loss += loss.item()
    print(f"Epoch {epoch + 1} - test loss: {test_loss / ev_length:.4f}")
    if (epoch + 1) % check_step == 0:
        torch.save(
            {
                "model": "Dual-Rnn-Model",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cost": test_loss / ev_length,
            },
            "./model/checkpoint.pt",
        )










''''train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

for epoch in range():

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
print(encode)'''