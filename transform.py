import torch
import numpy as np


def make_joint_vector(read_data):
    data = torch.cat(read_data, 0)
    data = data.view(-1, 14, 3)
    frame_size = data.shape[0]

    assert frame_size >= 2, "not_enough frame!"

    out_data = torch.tensor(data[1] - data[0])
    out_data = out_data.unsqueeze(0)
    for i in range(2, frame_size):
        out_data = torch.cat([out_data, (data[i] - data[i-1]).unsqueeze(0)], dim=0)
    out_data = torch.nn.functional.normalize(out_data , dim = 0)
    return out_data

