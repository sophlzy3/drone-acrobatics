import torch

def add_gaussian_noise(x, mean=0.0, std=0.1):
    """
    Adds Gaussian noise to a tensor.

    Parameters:
    x (torch.Tensor): Input tensor.
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    torch.Tensor: Noisy tensor.
    """
    # Generate Gaussian noise
    noise = torch.randn_like(x) * std + mean
    return x + noise
