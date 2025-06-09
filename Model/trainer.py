"""
Training module for the transformer model.

This module provides a Trainer class that handles the training process
for the transformer encoder and MLP model combination.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time

class Trainer:
    """
    Trainer class for training the transformer model.
    
    This class handles the training process, including:
    - Training loop
    - Loss computation
    - Model evaluation
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): The model to train (transformer + MLP)
            criterion (nn.Module): Loss function
            optimizer (torch.optim.Optimizer): Optimizer
            device (str): Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        scheduler: Optional[Any] = None
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            num_epochs (int): Number of epochs to train
            scheduler (Any, optional): Learning rate scheduler
            
        Returns:
            Dict[str, list]: Training history containing losses
        """
        # Initialize history
        history = {
            'train_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update training loss
                train_loss += loss.item()
                
                # Print progress
                if batch_idx % 40 == 0:
                    print(f'Epoch: {epoch+1}/{num_epochs} '
                          f'Batch: {batch_idx}/{len(train_loader)} '
                          f'Loss: {loss.item():.4f}')
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            print(f'Epoch: {epoch+1}/{num_epochs} '
                  f'Train Loss: {train_loss:.4f} '
                  f'Time: {time.time() - start_time:.2f}s')
            
            # Update learning rate
            if scheduler:
                scheduler.step(train_loss)  # Pass training loss as metric
        
        return history
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update validation loss
                val_loss += loss.item()
        
        return val_loss / len(val_loader) 