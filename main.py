import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import train
from training.model import GRUMLPModel
from training.load_data import load_and_split

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
# train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
#      csv_path='data/train.csv', 
#      target_column=['angular_rates.x','angular_rates.y','angular_rates.z','thrust.x','thrust.y','thrust.z']   # columns for model output 
# )

train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
     features_csv_path='data/train_x.csv', 
     labels_csv_path='data/train_y.csv'
)

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ====== TRAIN =======
train(model, train_loader, val_loader, lr=learning_rate, num_epochs=num_epochs, plotting=True)