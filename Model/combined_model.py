import torch
import torch.nn as nn

class CombinedModel(nn.Module):
    def __init__(self, transformer, mlp):
        super().__init__()
        self.transformer = transformer
        self.mlp = mlp
        
    def forward(self, x):

        # Create padding mask (True where padding)
        padding_mask = (x == self.transformer.padding_token)  # [Batch_size, seq_len]
        
        # Pass through transformer
        transformer_output = self.transformer(x)  # Shape: [batch_size, seq_len, d_model]

        # Invert mask: 1 for real tokens, 0 for pad
        attention_mask = (~padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
       # Apply mask
        masked_output = transformer_output * attention_mask  # [batch_size, seq_len, d_model]

        # Sum and normalize
        summed = masked_output.sum(dim=1)  # [batch_size, d_model]
        count = attention_mask.sum(dim=1)  # [batch_size, 1]
        pooled_output = summed / count.clamp(min=1.0)  # Avoid division by zero
        
        # Pass through MLP
        output = self.mlp(pooled_output)  # Shape: [batch_size, 1]
        
        # Squeeze the last dimension to match target shape
        return output.squeeze(-1)  # Shape: [batch_size] 


