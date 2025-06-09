import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class TargetTransformer:
    """
    Class for transforming target values based on input sequence characteristics.
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def per_length(self, X: torch.Tensor, Y: torch.Tensor, e: int = 0) -> torch.Tensor:
        """
        Transform target values by dividing by sequence length plus epsilon.
        
        Args:
            X (torch.Tensor): Input tensor with padding (0)
            Y (torch.Tensor): Target values
            e (int): Epsilon value to add to length (default: 0)
            
        Returns:
            torch.Tensor: Transformed target values
        """
        # Calculate non-padding lengths for each sequence
        lengths = (X != 0).sum(dim=1).float()
        
        # Add epsilon to lengths
        lengths = lengths + e
        
        # Divide Y by lengths
        Y_transformed = Y / lengths
        
        return Y_transformed
    
    def per_length_standard(self, X: torch.Tensor, Y: torch.Tensor, e: int = 0) -> torch.Tensor:
        """
        Transform target values by per_length method and then standardize.
        
        Args:
            X (torch.Tensor): Input tensor with padding (0)
            Y (torch.Tensor): Target values
            e (int): Epsilon value to add to length (default: 0)
            
        Returns:
            torch.Tensor: Transformed and standardized target values
        """
        # First apply per_length transformation
        Y_transformed = self.per_length(X, Y, e)
        
        # Convert to numpy for sklearn
        Y_np = Y_transformed.cpu().numpy().reshape(-1, 1)
        
        # Fit and transform if not already fitted
        if not self.is_fitted:
            Y_np = self.scaler.fit_transform(Y_np)
            self.is_fitted = True
        else:
            Y_np = self.scaler.transform(Y_np)
        
        # Convert back to tensor
        Y_transformed = torch.from_numpy(Y_np).squeeze(-1)
        
        return Y_transformed
    
    def inverse_per_length(self, X: torch.Tensor, Y_transformed: torch.Tensor, e: int = 0) -> torch.Tensor:
        """
        Inverse transform the per_length transformation.
        
        Args:
            X (torch.Tensor): Input tensor with padding (0)
            Y_transformed (torch.Tensor): Transformed target values
            e (int): Epsilon value that was added to length (default: 0)
            
        Returns:
            torch.Tensor: Original scale target values
        """
        # Calculate lengths (same as in per_length)
        lengths = (X != 0).sum(dim=1).float()
        lengths = lengths + e
        
        # Multiply by lengths to get original values
        Y_original = Y_transformed * lengths
        
        return Y_original
    
    def inverse_per_length_standard(self, X: torch.Tensor, Y_transformed: torch.Tensor, e: int = 0) -> torch.Tensor:
        """
        Inverse transform the per_length_standard transformation.
        
        Args:
            X (torch.Tensor): Input tensor with padding (0)
            Y_transformed (torch.Tensor): Transformed and standardized target values
            e (int): Epsilon value that was added to length (default: 0)
            
        Returns:
            torch.Tensor: Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse transform")
            
        # First inverse the standardization
        Y_np = Y_transformed.cpu().numpy().reshape(-1, 1)
        Y_unstandardized = self.scaler.inverse_transform(Y_np)
        Y_unstandardized = torch.from_numpy(Y_unstandardized).squeeze(-1)
        
        # Then inverse the per_length transformation
        Y_original = self.inverse_per_length(X, Y_unstandardized, e)
        
        return Y_original 