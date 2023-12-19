import torch
import numpy as np

max_frame = 12
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def make_joint_vector(read_data, norm=True):
    data = torch.tensor(read_data).to(device = device, dtype=torch.float32)
    data = data.view(-1, 51)
    frame_size = data.shape[0]

    assert frame_size >= 2, "not_enough frame!"

    out_data = data[1] - data[0]
    out_data = out_data.unsqueeze(0)
    for i in range(2, frame_size):
        out_data = torch.cat([out_data, (data[i] - data[i-1]).unsqueeze(0)], dim=0)
    if norm:
        out_data = torch.nn.functional.normalize(out_data , dim = 0)
    return out_data

def data_bind(data1, data2):
    out_data = torch.cat([data1.unsqueeze(0) if data1.dim() == 2 else data1, data2.unsqueeze(0) if data2.dim() == 2 else data2], dim=0)
    return out_data

