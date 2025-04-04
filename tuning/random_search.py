def random_search_training(model_fn, dataset, param_grid, num_trials=5, device='cuda'):
    ''' 
    Random search takes in a parameter grid consisting of batch size, learning rate and epochs, 
    as well as the model function, and randomly selects hyperparameters. After training with each combination, 
    it returns the best configuration of hyperparameters. 
    '''
    
    best_config = None
    best_model = None
    best_loss = float('inf')

    for trial in range(num_trials):
        # Randomly sample parameters
        config = {
            'batch_size': random.choice(param_grid['batch_size']),
            'lr': random.choice(param_grid['learning_rate']),
            'epochs': random.choice(param_grid['epochs'])
        }

        # Setup DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

        # Train model
        model = model_fn()
        model = train_model(model, loader, config['lr'], config['epochs'], device=device)

        # Evaluate (quick average loss on a few batches)
        model.eval()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                total_loss += criterion(out, yb).item()
                break  # only test on 1 batch for speed
        avg_loss = total_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model = model
            best_config = config

        print(f"Trial {trial+1}: {config} → Loss: {avg_loss:.4f}")

    return best_model, best_config

'''
example grid: 
param_grid = {
    'batch_size': [16, 32, 64],
    'learning_rate': [0.1, 0.01, 0.001],
    'epochs': [5, 10, 20]
}
'''

