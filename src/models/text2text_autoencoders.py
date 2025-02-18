# src/models/text2text_autoencoders.py

import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        mode: str = 'fixed',
        fixed_scale: float = 1.0, 
        learned_scale: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.mode = mode.lower()
        self.fixed_scale = fixed_scale
        self.learned_scale = learned_scale

        if self.mode in ['fixed', 'both']:
            pe = torch.zeros(1, max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        if self.mode in ['learnable', 'both']:
            self.learned_pe = nn.Embedding(max_len, d_model)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.size()) != 3:
            raise ValueError(f"Expected 3D tensor (batch_size, seq_len, d_model), got {x.size()}")
            
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        if self.mode == 'fixed':
            return x + self.fixed_scale * self.pe[:, :seq_len, :]
        elif self.mode == 'learnable':
            positions = torch.arange(seq_len, device=x.device).expand(x.size(0), -1)
            return x + self.learned_scale * self.learned_pe(positions)
        elif self.mode == 'both':
            fixed = self.fixed_scale * self.pe[:, :seq_len, :]
            positions = torch.arange(seq_len, device=x.device).expand(x.size(0), -1)
            learned = self.learned_scale * self.learned_pe(positions)
            return x + fixed + learned
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

class PositionalAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        target_seq_len: int = 3,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_len: int = 5000,
        pe_mode: str = 'fixed',
        use_normalization: bool = True,
        norm_type: str = 'batch',
        fixed_scale: float = 1.0,
        learned_scale: float = 1.0
    ):
        super().__init__()
        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.target_seq_len = target_seq_len
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.use_normalization = use_normalization
        self.norm_type = norm_type

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            mode=pe_mode,
            fixed_scale=fixed_scale,
            learned_scale=learned_scale
        )

        # Normalization factory
        def get_norm_layer(dim):
            if not use_normalization:
                return nn.Identity()
            if norm_type == 'batch':
                return nn.BatchNorm1d(dim)
            elif norm_type == 'layer':
                return nn.LayerNorm(dim)
            else:
                raise ValueError(f"Invalid norm_type: {norm_type}")

        # Encoder layers
        encoder_layers = []
        current_dim = d_model
        for _ in range(num_encoder_layers):
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                get_norm_layer(hidden_dim),
                getattr(nn, activation)(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Bottleneck - adjusted for target sequence length
        bottleneck_dim = hidden_dim // 2
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim * target_seq_len),
            get_norm_layer(bottleneck_dim * target_seq_len),
            getattr(nn, activation)(),
            nn.Dropout(dropout)
        )

        # Decoder layers
        decoder_layers = []
        current_dim = bottleneck_dim
        for i in range(num_encoder_layers):
            out_dim = d_model if i == num_encoder_layers - 1 else hidden_dim
            decoder_layers.extend([
                nn.Linear(current_dim, out_dim),
                get_norm_layer(out_dim),
                getattr(nn, activation)(),
                nn.Dropout(dropout)
            ])
            current_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)

        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, target_seq_len, vocab_size)
        """
        # Input shape check
        if len(src.shape) != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, seq_len), got shape {src.shape}")
        
        batch_size = src.size(0)
        
        # Truncate input if needed (keeping right side for source)
        if src.size(1) > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for encoder
        seq_len = x.size(1)
        x = x.reshape(-1, self.d_model)  # (batch_size * seq_len, d_model)
        
        # Encode
        x = self.encoder(x)  # (batch_size * seq_len, hidden_dim)
        
        # Reshape for bottleneck
        x = x.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Bottleneck
        x = self.bottleneck(x)  # (batch_size, bottleneck_dim * target_seq_len)
        x = x.reshape(batch_size, self.target_seq_len, self.bottleneck_dim)  # (batch_size, target_seq_len, bottleneck_dim)
        
        # Decode
        x = x.reshape(-1, self.bottleneck_dim)  # (batch_size * target_seq_len, bottleneck_dim)
        x = self.decoder(x)  # (batch_size * target_seq_len, d_model)
        
        # Reshape back
        x = x.reshape(batch_size, self.target_seq_len, self.d_model)  # (batch_size, target_seq_len, d_model)
        
        # Final projection to vocabulary
        logits = self.fc(x)  # (batch_size, target_seq_len, vocab_size)
        
        return logits
