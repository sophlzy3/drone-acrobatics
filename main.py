import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import train
from training.model import GRUMLPModel

# ====== MODEL =======
input_size = 10        # Input feature size
hidden_size = 64       # GRU hidden size
num_gru_layers = 2     # Number of GRU layers
mlp_hidden_size = 32   # Hidden size for MLP
output_size = 1        # Final output size
seq_len = 20           # Sequence length
# ====================
learning_rate = 1e-3
num_epochs = 10
batch_size = 32
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUMLPModel(input_size, hidden_size, num_gru_layers, mlp_hidden_size, output_size).to(device)

# ====== DATA =======
X_train = torch.randn(1000, seq_len, input_size)
y_train = torch.randn(1000, output_size)
X_val = torch.randn(200, seq_len, input_size)
y_val = torch.randn(200, output_size)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ====== TRAIN =======
train(model, train_loader, val_loader, lr=learning_rate, num_epochs=num_epochs, plotting=True)