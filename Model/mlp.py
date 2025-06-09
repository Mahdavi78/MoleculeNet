"""
MLP (Multi-Layer Perceptron) implementation for processing transformer outputs.

This module provides structure for MLP that can process outputs from
the transformer encoder. The specific architecture will be implemented later.
"""

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for processing transformer outputs.
    
    This class provides a basic structure. It's designed to work with the
    transformer encoder outputs.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dims: List[int] = [512, 256], activations: List[str] = ['relu', 'relu']):
        """
        Initialize MLP with configurable dimensions and activations.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            dims (List[int]): List of hidden layer dimensions
            activations (List[str]): List of activation functions ('relu', 'leaky_relu', 'sigmoid', 'tanh', 'none', '')
        """
        super().__init__()
        
        # Validate inputs
        if len(dims) != len(activations):
            raise ValueError("Number of dimensions must match number of activations")
            
        # Create layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim, activation in zip(dims, activations):
            layers.append(nn.Linear(prev_dim, dim))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif activation.lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() not in ['none', '']:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x) 