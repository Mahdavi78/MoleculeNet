"""
Transformer Encoder implementation for molecular sequence processing.

This module implements a transformer encoder architecture that can be used for
processing molecular sequences (SMILES) and learning molecular representations.
"""

import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder implementation for molecular sequence processing.
    
    This class implements a transformer encoder architecture with:
    - Multi-head self-attention
    - Position-wise feed-forward networks
    - Layer normalization
    - Residual connections
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        padding_token: int = 0
    ):
        """
        Initialize the Transformer Encoder.
        
        Args:
            vocab_size (int): Size of the vocabulary
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout probability
            max_seq_length (int): Maximum sequence length
            padding_token (int): Token value used for padding (default: 0)
        """
        super().__init__()
        
        # Store padding token value
        self.padding_token = padding_token
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters of the model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len)
                               where padding_token values indicate padding
            src_mask (torch.Tensor, optional): Mask for the attention mechanism
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Create padding mask from input (True for padding tokens)
        src_key_padding_mask = (src == self.padding_token)
        
        # Token embedding
        x = self.token_embedding(src)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        output = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    
    This class implements the sinusoidal positional encoding described in
    "Attention Is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Dimension of the model
            dropout (float): Dropout probability
            max_len (int): Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 