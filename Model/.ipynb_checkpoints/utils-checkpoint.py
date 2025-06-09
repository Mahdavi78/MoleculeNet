import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_data_loaders(X, Y, p=0.1, split_ratios=[0.15, 0.15, 0.7], batch_size=32):
    """
    Create data loaders with test set having least padding.
    
    Args:
        X (torch.Tensor): Input tensor
        Y (torch.Tensor): Target tensor
        p (float): Percentage of data to use (default: 0.1)
        split_ratios (list): List of ratios for [test, val, train] (default: [0.15, 0.15, 0.7])
        batch_size (int): Batch size for data loaders (default: 32)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Calculate padding counts and sort indices
    padding_counts = (X == 0).sum(dim=1)
    sorted_indices = torch.argsort(padding_counts)
    
    # Extract p percent of data
    total_size = len(X)
    subset_size = int(p * total_size)
    indices = sorted_indices[:subset_size]
    X_subset = X[indices]
    Y_subset = Y[indices]
    
    # Split subset into two parts based on padding
    half_size = subset_size // 2
    less_padded_X = X_subset[:half_size]
    less_padded_Y = Y_subset[:half_size]
    
    # Calculate split sizes
    test_size = int(split_ratios[0] * half_size)
    val_size = int(split_ratios[1] * half_size)
    train_size = half_size - test_size - val_size
    
    # Create datasets
    train_dataset = TensorDataset(
        less_padded_X[:train_size], 
        less_padded_Y[:train_size]
    )
    val_dataset = TensorDataset(
        less_padded_X[train_size:train_size+val_size], 
        less_padded_Y[train_size:train_size+val_size]
    )
    test_dataset = TensorDataset(
        less_padded_X[train_size+val_size:], 
        less_padded_Y[train_size+val_size:]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Print statistics
    print(f"Data split statistics:")
    print(f"Total data size: {total_size}")
    print(f"Subset size ({p*100}%): {subset_size}")
    print(f"Less padded part size: {half_size}")
    print(f"\nSplit sizes:")
    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    print(f"\nPadding statistics per split:")
    print(f"Train set avg padding: {(less_padded_X[:train_size] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Val set avg padding: {(less_padded_X[train_size:train_size+val_size] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Test set avg padding: {(less_padded_X[train_size+val_size:] == 0).sum(dim=1).float().mean():.2f}")
    
    return train_loader, val_loader, test_loader 