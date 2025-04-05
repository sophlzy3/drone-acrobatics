import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math

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

def normalize(x):
    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0]
    # Handle the case where min == max (avoid division by zero)
    denom = x_max - x_min
    denom[denom == 0] = 1.0
    return 2 * (x - x_min) / denom - 1

def reshape_data(train_x, train_y, val_x, val_y, test_x, test_y, seq_len):
    # Create sequences for training data
    train_x_seq = create_sequences(train_x, seq_len)
    train_y_seq = train_y[seq_len-1:]

    # Create sequences for validation data
    val_x_seq = create_sequences(val_x, seq_len)
    val_y_seq = val_y[seq_len-1:]

    # Create sequences for test data
    test_x_seq = create_sequences(test_x, seq_len)
    test_y_seq = test_y[seq_len-1:]

    # print(f"Reshaped train_x shape: {train_x_seq.shape}")
    # print(f"Reshaped train_y shape: {train_y_seq.shape}")
    # print(f"Reshaped val_x shape: {val_x_seq.shape}")
    # print(f"Reshaped val_y shape: {val_y_seq.shape}")
    # print(f"Reshaped test_x shape: {test_x_seq.shape}")
    # print(f"Reshaped test_y shape: {test_y_seq.shape}")

    return train_x_seq, train_y_seq, val_x_seq, val_y_seq, test_x_seq, test_y_seq

# Reshape data for GRU model (batch, seq_len, features)
# We'll create sequences by sliding a window of seq_len over the data
def create_sequences(x, seq_length):
    sequences = []
    for i in range(len(x) - seq_length + 1):
        sequences.append(x[i:i+seq_length])
    return torch.stack(sequences)

def train(model, train_loader, val_loader, lr, num_epochs, plotting=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Add gradient clipping to prevent exploding gradients
    clip_value = 1.0
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            # Check for NaN values in the batch
            if torch.isnan(xb).any() or torch.isnan(yb).any():
                print("Warning: NaN values found in input data")
                continue
                
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            
            # Check if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected in epoch {epoch+1}")
                continue
                
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
            running_loss += loss.item()
        
        # Skip NaN loss epochs
        if not math.isnan(running_loss / len(train_loader)):
            train_losses.append(running_loss / len(train_loader))
        else:
            print(f"Epoch {epoch+1}: NaN train loss - skipping")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    if plotting:
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.show()
    return train_losses, val_losses

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
