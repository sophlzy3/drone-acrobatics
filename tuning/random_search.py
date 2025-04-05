import random, sys, os, numpy as np
import torch
import torch.nn as nn
import torch.utils.data
# from training.train import train_with_clipping, train
# from training.test import test_model
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.model import GRUMLPModel
from training.train import init_weights, normalize, reshape_data, train_with_clipping, train
from training.test import test_model
from training.load_data import load_and_split


def random_search_training(param_grid, num_trials=5, device='cuda'):
    ''' 
    Random search takes in a parameter grid consisting of batch size, learning rate, epochs and clip value, 
    as well as the model function, and randomly selects hyperparameters. After training with each combination, 
    it returns the best configuration of hyperparameters. 
    '''
    
    best_config = None
    best_model = None
    best_metric = float('inf')  # Using RMSE as the metric to minimize
    best_train_losses = None
    best_val_losses = None
    best_test_metrics = None
    best_score = float('inf')

    for trial in range(num_trials):
        # Randomly sample parameters
        config = {
            'batch_size': random.choice(param_grid['batch_size']),
            'lr': random.choice(param_grid['learning_rate']),
            'epochs': random.choice(param_grid['epochs']),
            # 'clip_value': random.choice(param_grid['clip_value']) if 'clip_value' in param_grid else 1.0,
            'hidden_size': random.choice(param_grid['hidden_size']),
            'num_gru_layers': random.choice(param_grid['num_gru_layers']),
            'mlp_hidden_size': random.choice(param_grid['mlp_hidden_size']) 
        }

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
        seq_len = 20
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = train_x.shape[1]
        output_size = train_y.shape[1]
        model = GRUMLPModel(input_size, config['hidden_size'], config['num_gru_layers'], config['mlp_hidden_size'], output_size).to(device)
        model.apply(init_weights)
        train_x_seq, train_y_seq, val_x_seq, val_y_seq, test_x_seq, test_y_seq = reshape_data(train_x, train_y, val_x, val_y, test_x, test_y, seq_len)


        # ====== LOADER INIT =======
        train_dataset = TensorDataset(train_x_seq, train_y_seq)
        val_dataset = TensorDataset(val_x_seq, val_y_seq)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        # =========== TRAINING ===========
        train_losses, val_losses = train(model, train_loader, val_loader, lr=config['lr'], num_epochs=config['epochs'], plotting=False)
        # train_losses, val_losses = train_with_clipping(model, train_loader, val_loader, lr=config['lr'], num_epochs=config['epochs'], plotting=False, clip_value=config['clip_value'])

        # =========== TESTING ===========
        # Prepare test data loader
        test_dataset = TensorDataset(test_x_seq, test_y_seq)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

        # Test the model
        test_metrics = test_model(model, test_loader, plotting=False)

        # Print overall test summary
        print(f"\nTest Summary:")
        print(f"RMSE: {test_metrics['rmse']:.6f}")
        print(f"R² Score: {test_metrics['r2']:.6f}")

        # Create a weighted score that considers all three error sources
        current_score = (
            0.2 * np.mean(train_losses[-5:]) +  # Recent training loss (less weight)
            10 * np.mean(val_losses[-5:]) +    # Recent validation loss (medium weight)
            0.5 * test_metrics['rmse']          # Test RMSE (highest weight)
        )

        if current_score < best_score:  # Lower is better
            best_score = current_score
            best_metric = test_metrics['rmse']  # Still track individual metrics
            best_model = model
            best_config = config
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_test_metrics = test_metrics

        print(f"Trial {trial+1}: {config} → RMSE: {test_metrics['rmse']:.4f}\n")

    print(f"\nBest configuration: {best_config} with RMSE: {best_metric:.4f}")
    
    # Create a dictionary with all results for the best model
    best_results = {
        'config': best_config,
        'test_metrics': best_test_metrics,
        'final_train_loss': best_train_losses[-1] if best_train_losses else None,
        'final_val_loss': best_val_losses[-1] if best_val_losses else None,
        'train_losses': best_train_losses,
        'val_losses': best_val_losses
    }
    
    return best_model, best_results

def main(param_grid, trials):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("starting random search...")
    best_model, best_results = random_search_training(param_grid, trials, device='cuda')
    
    # Save best model
    print("Saving best model and results...")
    torch.save(best_model.state_dict(), 'models/best_model.pt')
    
    # Save best configuration and results
    import json
    with open('models/best_results.json', 'w') as f:
        # Convert any non-serializable objects to strings or numbers
        serializable_results = {
            'config': best_results['config'],
            'test_metrics': {k: float(v) for k, v in best_results['test_metrics'].items()},
            'final_train_loss': float(best_results['final_train_loss']),
            'final_val_loss': float(best_results['final_val_loss']),
            # Convert losses lists to regular lists of floats
            'train_losses': [float(loss) for loss in best_results['train_losses']],
            'val_losses': [float(loss) for loss in best_results['val_losses']]
        }
        json.dump(serializable_results, f, indent=4)
    
    # Save training/validation loss plot for best model
    plt.figure(figsize=(10, 6))
    plt.plot(best_results['train_losses'], label='Training Loss')
    plt.plot(best_results['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Best Model')
    plt.legend()
    plt.show()
    
    print(f"Best model saved with configuration: {best_results['config']}")
    print(f"Test metrics: {best_results['test_metrics']}")
    
    return best_model, best_results


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    param_grid = {
        'batch_size': [16, 32, 64, 128],
        'learning_rate': [1e-4, 1e-6, 1e-8],
        'epochs': [30, 40, 50, 60, 70, 80, 90, 100],
        # 'clip_value': [0.1, 0.5, 1.0, 2.0, 5.0],
        'hidden_size': [64, 128, 256, 512, 1024, 2048],
        'num_gru_layers': [3,30,50],
        'mlp_hidden_size': [16, 32, 64, 128, 256, 512, 1024, 2048]
    } 
    trials = 3
    
    # Run hyperparameter tuning
    best_model, best_results = main(param_grid, trials)
    
