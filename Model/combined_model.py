import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, transformer, mlp):
        super().__init__()
        self.transformer = transformer
        self.mlp = mlp
        
    def forward(self, x):
        # Pass through transformer
        transformer_output = self.transformer(x)  # Shape: [batch_size, seq_len, d_model]
        
        # Take mean across sequence length dimension
        pooled_output = transformer_output.mean(dim=1)  # Shape: [batch_size, d_model]
        
        # Pass through MLP
        output = self.mlp(pooled_output)  # Shape: [batch_size, 1]
        
        # Squeeze the last dimension to match target shape
        return output.squeeze(-1)  # Shape: [batch_size] 