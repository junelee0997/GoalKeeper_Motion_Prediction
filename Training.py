import torch
import torch.nn as nn
import numpy as np
import New_Seq2Seq_Model as NS
import data_util as du
import transform as tf
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

rnn_size = 1024
human_size = 42
max_norm = 5
batch_size = 16
data_count = 1000
iterations = 100000
check_step = 1000

goal_dataset = du.data_splice()
#goal_train, kick_train = du.data_splice(ver=True)
goal_train = du.data_splice()
goal_dataset = tf.make_joint_vector(goal_dataset)
goal_train = tf.make_joint_vector(goal_train)
for i in range(1, data_count):
    ngoal_dataset = du.data_splice(f'data/{i}.txt')
    ngoal_train = du.data_splice(f'data/test/{i}.txt')
    goal_train = tf.data_bind(goal_train, ngoal_train)
    goal_dataset = tf.data_bind(goal_dataset, ngoal_dataset)
    

dataset = TensorDataset(goal_dataset, goal_train)
dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
goal_encode = NS.Encoder(human_size, rnn_size, 1)
kick_encode = NS.Encoder(human_size, rnn_size, 1)
decode = NS.Decoder(human_size, rnn_size, 1)
model = NS.Seq2Seq(goal_encode, decode, kick_encode, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.MSELoss().to(device)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=160000 // data_count,gamma=0.95)

epochs = batch_size * iterations // data_count

for epoch in range(epochs):
    model.train()
    for x, t in enumerate(dataset):
        y, z = t
        prediction = model(y, z)
        loss = criterion(prediction, z).to(device, dtype=torch.float32)
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, epochs, x + 1, len(dataset),
            loss.item()
        ))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
    scheduler.step()








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