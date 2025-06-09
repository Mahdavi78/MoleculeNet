import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_data_loaders(X, Y, p=0.1, split_ratios=[0.15, 0.15, 0.7], batch_size=32):
    """
    Create data loaders with test set having least padding.
    Test set is selected from top 30% of sequences with least padding.
    
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
    
    # Calculate sizes
    total_size = len(X)
    subset_size = int(p * total_size)
    test_selection_size = int(0.4 * total_size)  # Top 40% with least padding
    
    # Get test indices from top 40% with least padding
    test_pool_indices = sorted_indices[:test_selection_size]
    test_size = int(split_ratios[0] * subset_size)
    test_selected = torch.randperm(len(test_pool_indices))[:test_size]
    test_indices = test_pool_indices[test_selected]
    
    # Get remaining indices for train and val
    remaining_indices = sorted_indices[test_selection_size:]
    remaining_size = subset_size - test_size
    remaining_selected = torch.randperm(len(remaining_indices))[:remaining_size]
    remaining_indices = remaining_indices[remaining_selected]
    
    # Calculate val and train sizes
    val_size = int(split_ratios[1] * subset_size)
    train_size = remaining_size - val_size
    
    # Create datasets
    test_dataset = TensorDataset(
        X[test_indices], 
        Y[test_indices]
    )
    val_dataset = TensorDataset(
        X[remaining_indices[:val_size]], 
        Y[remaining_indices[:val_size]]
    )
    train_dataset = TensorDataset(
        X[remaining_indices[val_size:]], 
        Y[remaining_indices[val_size:]]
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
    print(f"Test selection pool size (30%): {test_selection_size}")
    print(f"\nSplit sizes:")
    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(val_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    print(f"\nPadding statistics per split:")
    print(f"Train set avg padding: {(X[remaining_indices[val_size:]] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Val set avg padding: {(X[remaining_indices[:val_size]] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Test set avg padding: {(X[test_indices] == 0).sum(dim=1).float().mean():.2f}")
    
    return train_loader, val_loader, test_loader 