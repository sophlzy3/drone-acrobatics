import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
def test_model(model, test_loader, scaler=None, plotting=True):
    """
    Test the model on the test dataset and return accuracy metrics.
    
    Args:
        model: The trained model
        test_loader: DataLoader with test data
        scaler: Optional scaler used for denormalizing predictions
        plotting: Whether to plot predictions vs actual values
        
    Returns:
        dict: Dictionary containing accuracy metrics (MSE, MAE, R^2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Prepare lists for collecting predictions and actual values
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Get predictions
            predictions = model(x_batch)
            
            # Store predictions and targets
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Print results
    print(f"\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # Optional: Plot predictions vs actual for each output variable
    if plotting:
        import matplotlib.pyplot as plt
        
        # Number of output variables
        n_outputs = all_targets.shape[1]
        
        # Plot each output variable separately
        plt.figure(figsize=(15, n_outputs * 4))
        
        for i in range(n_outputs):
            plt.subplot(n_outputs, 1, i+1)
            
            # Select first 100 samples for clearer visualization
            samples_to_show = min(100, len(all_preds))
            
            plt.plot(range(samples_to_show), all_targets[:samples_to_show, i], 'b-', label='Actual')
            plt.plot(range(samples_to_show), all_preds[:samples_to_show, i], 'r-', label='Predicted')
            
            plt.title(f'Output Variable {i+1}')
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Return metrics as a dictionary
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
