import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import train, train_with_clipping, init_weights, normalize, reshape_data
from training.model import GRUMLPModel
from training.load_data import load_and_split
from training.test import test_model

# ====== DATA =======
train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
     features_csv_path='data/train_x.csv', 
     labels_csv_path='data/train_y.csv'
)
train_x = torch.nan_to_num(train_x, nan=0.0)
train_y = torch.nan_to_num(train_y, nan=0.0)
val_x = torch.nan_to_num(val_x, nan=0.0)
val_y = torch.nan_to_num(val_y, nan=0.0)
test_x = torch.nan_to_num(test_x, nan=0.0)
test_y = torch.nan_to_num(test_y, nan=0.0)

train_x = normalize(train_x)
# train_y = normalize(train_y)

# ====== MODEL INIT =======
input_size = train_x.shape[1]  # Input feature size (number of columns)
hidden_size = 64       # GRU hidden size
num_gru_layers = 3     # Number of GRU layers
mlp_hidden_size = 32   # Hidden size for MLP
output_size = train_y.shape[1]  # Output size (number of columns in train_y)
seq_len = 20           # Sequence length
# ====================
learning_rate = 0.0001
num_epochs = 20
batch_size = 16
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUMLPModel(input_size, hidden_size, num_gru_layers, mlp_hidden_size, output_size).to(device)
model.apply(init_weights)
train_x_seq, train_y_seq, val_x_seq, val_y_seq, test_x_seq, test_y_seq = reshape_data(train_x, train_y, val_x, val_y, test_x, test_y, seq_len)

# print(f'train_x shape: {train_x.shape}')
# print(f'train_y shape: {train_y.shape}')
# print(f"Reshaped train_x shape: {train_x_seq.shape}")
# print(f"Reshaped train_y shape: {train_y_seq.shape}")

# ====== LOADER INIT =======
train_dataset = TensorDataset(train_x_seq, train_y_seq)
val_dataset = TensorDataset(val_x_seq, val_y_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# =========== TRAINING ===========
# train(model, train_loader, val_loader, lr=learning_rate, num_epochs=num_epochs, plotting=True)
train_with_clipping(model, train_loader, val_loader, lr=learning_rate, num_epochs=num_epochs, plotting=True, clip_value=0.5)

# =========== TESTING ===========
# Prepare test data loader
test_dataset = TensorDataset(test_x_seq, test_y_seq)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Test the model
test_metrics = test_model(model, test_loader, plotting=True)

# Print overall test summary
print(f"\nTest Summary:")
print(f"RMSE: {test_metrics['rmse']:.6f}")
print(f"R² Score: {test_metrics['r2']:.6f}")
