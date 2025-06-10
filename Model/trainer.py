"""
Training module for the transformer model.

This module provides a Trainer class that handles the training process
for the transformer encoder and MLP model combination.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
import time
import os

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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = 'Experiments/experiment'
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

        #initialize TensorBoard writer

        # Create the directory if it doesn't exist
        if not os.path.exists(log_dir):
            print(f"not found directory, creating directory: {log_dir}")
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        scheduler: Optional[Any] = None,
        val_loader: Optional[DataLoader] = None
    ) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            num_epochs (int): Number of epochs to train
            scheduler (Any, optional): Learning rate scheduler
            val_loader (DataLoader, optional): Validation data loader
            
        Returns:
            Dict[str, list]: Training history containing losses
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [] if val_loader else None
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
                
                # Log batch loss to TensorBoard
                self.writer.add_scalar('Loss/train_batch', loss.item(), 
                                       epoch * len(train_loader) + batch_idx)
                
                # Print progress
                if batch_idx % 40 == 0:
                    print(f'Epoch: {epoch+1}/{num_epochs} '
                          f'Batch: {batch_idx}/{len(train_loader)} '
                          f'Loss: {loss.item():.4f}')
                    
                    # Log gradients for each parameter
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad.data,
                                                       epoch * len(train_loader) + batch_idx)
                    

            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # Log epoch training loss
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            print(f'Epoch: {epoch+1}/{num_epochs} '
                  f'Train Loss: {train_loss:.4f} '
                  f'Time: {time.time() - start_time:.2f}s')
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']  # Assuming a single optimizer
            self.writer.add_scalar('Learning_rate',
                                    current_lr, epoch * len(train_loader) + batch_idx)
            
            # Log weight distributions for each parameter
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
            
            
            # Validation phase
            if val_loader:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Log validation loss
                self.writer.add_scalar('Loss/validation', val_loss, epoch)
                print(f'Epoch: {epoch+1}/{num_epochs} Validation Loss: {val_loss:.4f}')
            
            # Update learning rate
            if scheduler:
                scheduler.step(train_loss)  # Pass training loss as metric
        
        # Close TensorBoard writer
        self.writer.close()

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