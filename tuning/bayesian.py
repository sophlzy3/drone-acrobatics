from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import torch
import torch.nn as nn
import torch.optim as optim

from train import train 

# Define the hyperparameter search space, can change this 
space = [
    Real(1e-4, 1e-1, name='lr', prior='log-uniform'),
    Integer(5, 20, name='epochs'),
]

def bayesian_tuner(model_fn, train_loader, val_loader, test_loader, n_calls=10, device='cuda'):
    best_model = None
    best_score = 0
    best_params = None

    @use_named_args(space)
    def objective(lr, epochs):
        model = model_fn()
        acc, trained_model = train_eval(model, train_loader, val_loader, lr, epochs, device)

        nonlocal best_model, best_score, best_params
        if acc > best_score:
            best_score = acc
            best_model = trained_model
            best_params = {'lr': lr, 'epochs': epochs}

        print(f"Trial: lr={lr:.5f}, epochs={epochs} → Val Acc = {acc:.4f}")
        return -acc  # We minimize negative accuracy

    print("Starting Bayesian Optimization...\n")
    gp_minimize(objective, space, n_calls=n_calls, random_state=42)

    # Final evaluation on test set
    best_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = best_model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    test_acc = correct / total
    print(f"\n Best Params: {best_params}")
    print(f" Validation Accuracy: {best_score:.4f}")
    print(f" Test Accuracy: {test_acc:.4f}")

    return best_model, best_params, best_score, test_acc
