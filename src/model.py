import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class GRUMLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers, mlp_hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_gru_layers,
                          batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size)
        )

    def forward(self, x):
        out, _ = self.gru(x)               # out: (batch, seq_len, hidden)
        out = out[:, -1, :]                # take last time step
        return self.mlp(out)
