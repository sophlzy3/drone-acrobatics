import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.train import train
from training.model import GRUMLPModel
from training.load_data import load_and_split

# ====== DATA =======
# Load data first to determine input_size
train_x, val_x, test_x, train_y, val_y, test_y, scaler, feature_names = load_and_split(
     features_csv_path='data/train_x.csv', 
     labels_csv_path='data/train_y.csv'
)

# Convert NumPy arrays to PyTorch tensors
train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_y)
val_x = torch.FloatTensor(val_x)
val_y = torch.FloatTensor(val_y)
test_x = torch.FloatTensor(test_x)
test_y = torch.FloatTensor(test_y)

# Print shapes for debugging
print(f"Original train_x shape: {train_x.shape}")
print(f"Original train_y shape: {train_y.shape}")

# ====== MODEL =======
input_size = train_x.shape[1]  # Input feature size (number of columns)
hidden_size = 64       # GRU hidden size
num_gru_layers = 2     # Number of GRU layers
mlp_hidden_size = 32   # Hidden size for MLP
output_size = train_y.shape[1]  # Output size (number of columns in train_y)
seq_len = 20           # Sequence length
# ====================
learning_rate = 1e-4   # Reduced learning rate
num_epochs = 10
batch_size = 32
# ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUMLPModel(input_size, hidden_size, num_gru_layers, mlp_hidden_size, output_size).to(device)

# Initialize weights with better defaults
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)

model.apply(init_weights)

# Reshape data for GRU model (batch, seq_len, features)
# We'll create sequences by sliding a window of seq_len over the data
def create_sequences(x, seq_length):
    sequences = []
    for i in range(len(x) - seq_length + 1):
        sequences.append(x[i:i+seq_length])
    return torch.stack(sequences)

# Create sequences for training data
train_x_seq = create_sequences(train_x, seq_len)
train_y_seq = train_y[seq_len-1:]

# Create sequences for validation data
val_x_seq = create_sequences(val_x, seq_len)
val_y_seq = val_y[seq_len-1:]

# Create sequences for test data
test_x_seq = create_sequences(test_x, seq_len)
test_y_seq = test_y[seq_len-1:]

print(f"Reshaped train_x shape: {train_x_seq.shape}")
print(f"Reshaped train_y shape: {train_y_seq.shape}")

# Create datasets
train_dataset = TensorDataset(train_x_seq, train_y_seq)
val_dataset = TensorDataset(val_x_seq, val_y_seq)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ====== TRAIN =======
# Modified train function with gradient clipping
def train_with_clipping(model, train_loader, val_loader, lr, num_epochs, plotting=True, clip_value=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if plotting:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.show()
    
    return train_losses, val_losses

# Use the modified train function with gradient clipping
train_with_clipping(model, train_loader, val_loader, lr=learning_rate, num_epochs=num_epochs, plotting=True, clip_value=1.0)