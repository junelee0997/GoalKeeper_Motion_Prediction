import numpy as np
import torch


def data_splice(directory='data/src.txt', ver = False):
    if ver:
        read_data = torch.from_numpy(np.genfromtxt(directory, delimiter=','))
        goal_read_data = ([read_data[i] for i in range(read_data.shape[0]) if i % 18 < 14 and i % 36 < 18])
        kick_read_data = ([read_data[i] for i in range(read_data.shape[0]) if i % 18 < 14 and i % 36 >= 18])
        return goal_read_data, kick_read_data
    else:
        read_data = torch.from_numpy(np.genfromtxt(directory, delimiter=','))
        out_data = ([read_data[i] for i in range(read_data.shape[0]) if i % 18 < 14])
        return out_data