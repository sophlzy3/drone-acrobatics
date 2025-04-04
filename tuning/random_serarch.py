import random
import torch
import torch.nn as nn
import torch.optim as optim

def random_search_tuner(model_fn, train_loader, val_loader=None, 
                        search_space=None, num_trials=10, device='cuda' if torch.cuda.is_available() else 'cpu',
                        epochs=5, verbose=True):
    """
    Performs random search hyperparameter tuning.
    
    Args:
        model_fn: function that returns a new model instance (e.g., lambda: MyModel())
        train_loader: DataLoader for training data
        val_loader: optional DataLoader for validation data
        search_space: dict of parameter name -> list of values to sample from
        num_trials: number of random trials to run
        device: 'cuda' or 'cpu'
        epochs: number of epochs per trial
        verbose: whether to print progress

    Returns:
        best_model: the model with the best performance
        best_config: the hyperparameter configuration for the best model
    """
    best_loss = float('inf')
    best_model = None
    best_config = None

    for trial in range(num_trials):
        # Sample random hyperparameters
        config = {k: random.choice(v) for k, v in search_space.items()}

        model = model_fn().to(device)
        optimizer = getattr(optim, config['optimizer'])(model.parameters(), lr=config['lr'])
        criterion = getattr(nn, config['loss'])()

        if verbose:
            print(f"\nTrial {trial+1}/{num_trials} - Config: {config}")

        # Training loop
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            if verbose:
                print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        # Evaluation (on val_loader if given)
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        else:
            val_loss = avg_loss  # fallback to training loss

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_config = config

    return best_model, best_config


