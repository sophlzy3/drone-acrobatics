import json
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_loss_from_json(json_path, save_path=None):
    """
    Plot training and validation loss from a saved results JSON file.
    
    Args:
        json_path: Path to the JSON file with loss data
        save_path: Optional path to save the plot (if None, will display plot)
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract training and validation losses
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    
    # Create the figure
    plt.figure(figsize=(12, 7))
    
    # Plot losses
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    
    # Add grid and styling
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    
    # Add title with model info
    architecture = results.get('architecture', {})
    config = results.get('config', {})
    title = f"Training and Validation Loss\n"
    if architecture:
        title += f"Architecture: {architecture['hidden_size']} hidden, {architecture['num_gru_layers']} GRU layers\n"
    if config:
        title += f"LR: {config.get('lr', 'N/A')}, Batch: {config.get('batch_size', 'N/A')}"
    
    plt.title(title, fontsize=14)
    
    # Add legend with metrics if available
    legend_text = ['Training Loss', 'Validation Loss']
    if 'test_metrics' in results:
        metrics = results['test_metrics']
        plt.figtext(0.5, 0.01, 
                   f"Test RMSE: {metrics.get('rmse', 'N/A'):.4f} | "
                   f"Test R²: {metrics.get('r2', 'N/A'):.4f} | "
                   f"Test MAE: {metrics.get('mae', 'N/A'):.4f}",
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.legend(legend_text, loc='upper right')
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Default paths
    json_path = 'models/best_results.json'
    save_path = 'models/loss_plot.png'
    
    # Allow command line arguments to specify paths
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    if len(sys.argv) > 2:
        save_path = sys.argv[2]
    
    plot_loss_from_json(json_path, save_path) 