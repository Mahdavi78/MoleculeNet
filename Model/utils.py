import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_train_test_loaders(X, Y, p=0.1, split_ratios=[0.2, 0.8], batch_size=32):
    """
    Create data loaders with test set having least padding.
    Test set is selected from top 30% of sequences with least padding.
    
    Args:
        X (torch.Tensor): Input tensor
        Y (torch.Tensor): Target tensor
        p (float): Percentage of data to use (default: 0.1)
        split_ratios (list): List of ratios for [test, train] (default: [0.2, 0.8])
        batch_size (int): Batch size for data loaders (default: 32)
        
    Returns:
        tuple: (train_loader, test_loader)
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
    
    # Get remaining indices for train
    remaining_indices = sorted_indices[test_selection_size:]
    train_size = subset_size - test_size
    train_selected = torch.randperm(len(remaining_indices))[:train_size]
    train_indices = remaining_indices[train_selected]
    
    # Create datasets
    test_dataset = TensorDataset(
        X[test_indices], 
        Y[test_indices]
    )
    train_dataset = TensorDataset(
        X[train_indices], 
        Y[train_indices]
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    print(f"Test selection pool size (40%): {test_selection_size}")
    print(f"\nSplit sizes:")
    print(f"Training set: {len(train_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    print(f"\nPadding statistics per split:")
    print(f"Train set avg padding: {(X[train_indices] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Test set avg padding: {(X[test_indices] == 0).sum(dim=1).float().mean():.2f}")
    
    return train_loader, test_loader

def create_train_val_loader(train_loader, val_ratio=0.2, batch_size=None):
    """
    Split a train_loader into train and validation loaders.
    
    Args:
        train_loader (DataLoader): Original training data loader
        val_ratio (float): Ratio of data to use for validation (default: 0.2)
        batch_size (int, optional): Batch size for the new loaders. If None, uses the original batch size.
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get the dataset from the original loader
    dataset = train_loader.dataset
    X = dataset.tensors[0]
    Y = dataset.tensors[1]
    
    # Calculate sizes
    total_size = len(X)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    
    # Create random indices for splitting
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(
        X[train_indices],
        Y[train_indices]
    )
    val_dataset = TensorDataset(
        X[val_indices],
        Y[val_indices]
    )
    
    # Use original batch size if not specified
    if batch_size is None:
        batch_size = train_loader.batch_size
    
    # Create new loaders
    new_train_loader = DataLoader(
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
    
    # Print statistics
    print(f"Train-Val Split Statistics:")
    print(f"Original train set size: {total_size}")
    print(f"New train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"\nPadding statistics per split:")
    print(f"Train set avg padding: {(X[train_indices] == 0).sum(dim=1).float().mean():.2f}")
    print(f"Val set avg padding: {(X[val_indices] == 0).sum(dim=1).float().mean():.2f}")
    
    return new_train_loader, val_loader

def create_Incremental_loaders(train_loader, n=4, batch_size=None):
    """
    Create n incremental loaders from sorted data (by padding amount).
    Data is sorted so that sequences with most padding come first.
    
    Args:
        train_loader (DataLoader): Original training data loader
        n (int): Number of loaders to create (default: 4)
        batch_size (int, optional): Batch size for new loaders. If None, uses original batch size.
        
    Returns:
        list: List of n DataLoaders, each containing a portion of the sorted data
    """
    # Get data from loader
    dataset = train_loader.dataset
    X = dataset.tensors[0]
    Y = dataset.tensors[1]
    
    # Calculate padding counts and sort indices (most padding first)
    padding_counts = (X == 0).sum(dim=1)
    sorted_indices = torch.argsort(padding_counts, descending=True)
    
    # Sort the data
    X_sorted = X[sorted_indices]
    Y_sorted = Y[sorted_indices]
    
    # Calculate size of each portion
    total_size = len(X)
    portion_size = total_size // n
    
    # Use original batch size if not specified
    if batch_size is None:
        batch_size = train_loader.batch_size
    
    # Create n loaders
    loaders = []
    for i in range(n):
        start_idx = i * portion_size
        end_idx = start_idx + portion_size if i < n-1 else total_size
        
        # Create dataset for this portion
        portion_dataset = TensorDataset(
            X_sorted[start_idx:end_idx],
            Y_sorted[start_idx:end_idx]
        )
        
        # Create loader
        loader = DataLoader(
            portion_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        loaders.append(loader)
        
        # Print statistics for this portion
        print(f"\nLoader {i+1} Statistics:")
        print(f"Size: {len(portion_dataset)}")
        print(f"Average padding: {(X_sorted[start_idx:end_idx] == 0).sum(dim=1).float().mean():.2f}")
    
    return loaders 