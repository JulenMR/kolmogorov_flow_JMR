import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.data import Dataset
import h5py

class KolmogorovDataset(Dataset):
    def __init__(self, h5_path, sim_indices, min_val, max_val):
        self.h5_path = h5_path
        self.sim_indices = sim_indices
        self.min = min_val
        self.max = max_val
        self.steps = 200
        self.file = None

    def __len__(self):
        return len(self.sim_indices)*self.steps
    
    def __getitem__(self, idx):
        
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            self.u_dataset = self.file["valid/u"]

        idx_sim_list = idx // self.steps
        t = idx % self.steps
        sim_num = self.sim_indices[idx_sim_list]

        x_data = self.u_dataset[sim_num, t, :, :]
        y_data = self.u_dataset[sim_num, t, :, :]

        x_normalized = (x_data - self.min) / (self.max - self.min)
        y_normalized = (y_data - self.min) / (self.max - self.min)

        x_tensor = torch.tensor(x_normalized, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(0)

        return x_tensor, y_tensor
