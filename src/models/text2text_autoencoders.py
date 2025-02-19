# src/models/text2text_autoencoders.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple
import numpy as np


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightNormConv1d(nn.Module):
    """Weight-normalized 1D convolution with proper causal initialization"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=0  # No built-in padding
        )
        
        # Initialize parameters
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.conv.bias, 0.0)
        
        # Weight normalization parameters
        with torch.no_grad():
            weight_flat = self.conv.weight.view(out_channels, -1)
            weight_norms = torch.norm(weight_flat, dim=1, p=2)
            self.scale = nn.Parameter(weight_norms.clone())
            self.conv.weight.data = F.normalize(weight_flat, dim=1).view_as(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute normalized weights
        weight_flat = self.conv.weight.view(self.conv.out_channels, -1)
        weight = self.scale.view(-1, 1, 1) * F.normalize(weight_flat, dim=1).view_as(self.conv.weight)
        
        return F.conv1d(
            x, 
            weight, 
            self.conv.bias, 
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

class ConvS2SBlock(nn.Module):
    """Causal convolutional block with GLU activation"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        layer_idx: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        
        # Convolution with weight normalization
        self.conv = WeightNormConv1d(
            in_channels=input_dim,
            out_channels=2 * output_dim,
            kernel_size=kernel_size
        )
        
        # Residual projection
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if self.residual:
            nn.init.xavier_uniform_(self.residual.weight)
            nn.init.constant_(self.residual.bias, 0.0)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # Apply causal padding
        x = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        x = self.conv(x).transpose(1, 2)
        
        # GLU activation
        x = F.glu(x, dim=-1)
        
        # Residual connection
        if self.residual:
            residual = self.residual(residual)
        x = (x + residual) * math.sqrt(self.num_layers)
        
        # Mask padding
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        return self.dropout(x[:, :seq_len, :])  # Strict length control
        
class ConvS2SAttention(nn.Module):
    """Multi-step attention as described in the paper"""
    def __init__(self, decoder_dim: int, encoder_dim: int, hidden_dim: int):
        super().__init__()
        
        # Projections for attention
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project decoder and encoder states
        decoder_hidden = self.decoder_proj(decoder_state)  # [B, T, H]
        encoder_hidden = self.encoder_proj(encoder_out)    # [B, S, H]
        
        # Calculate attention scores
        # Add dims for broadcasting
        decoder_hidden = decoder_hidden.unsqueeze(2)  # [B, T, 1, H]
        encoder_hidden = encoder_hidden.unsqueeze(1)  # [B, 1, S, H]
        
        # Combined hidden
        combined = torch.tanh(decoder_hidden + encoder_hidden)  # [B, T, S, H]
        
        # Calculate attention scores
        attn_scores = self.output_proj(combined).squeeze(-1)  # [B, T, S]
        
        # Apply mask if provided
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                encoder_padding_mask.unsqueeze(1),
                float('-inf')
            )
        
        # Normalize scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Calculate context vectors
        context = torch.bmm(attn_weights, encoder_out)  # [B, T, E]
        
        return context, attn_weights

class ConvS2SEncoder(nn.Module):
    """Convolutional encoder"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_positions: int = 512
    ):
        super().__init__()
        self.max_positions = max_positions  # Add this line
        # Word embeddings
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings (learned, not sinusoidal)
        self.embed_positions = nn.Embedding(max_positions, embed_dim)
        
        # Register position indices buffer
        positions = torch.arange(max_positions).unsqueeze(0)
        self.register_buffer('positions', positions)
        
        # Convolutional layers
        self.layers = nn.ModuleList([
            ConvS2SBlock(
                input_dim=embed_dim if i == 0 else hidden_dim,
                output_dim=hidden_dim,
                kernel_size=kernel_size,
                layer_idx=i,
                num_layers=num_layers,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Output projection if needed
        self.output_projection = (
            nn.Linear(hidden_dim, embed_dim)
            if hidden_dim != embed_dim
            else None
        )
        
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get sequence positions
        positions = self.positions[:, :src_tokens.size(1)]
        
        # Combine token and position embeddings
        x = self.embed_tokens(src_tokens) + self.embed_positions(positions)
        
        # Apply convolutional layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # Final projection if needed
        if self.output_projection is not None:
            x = self.output_projection(x)
            
        return x

class ConvS2SDecoder(nn.Module):
    """Convolutional decoder with strict sequence length enforcement"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_positions: int = 512
    ):
        super().__init__()
        self.max_positions = max_positions
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.embed_positions = nn.Embedding(max_positions, embed_dim)
        self.register_buffer('position_ids', torch.arange(max_positions).unsqueeze(0))
        
        # Layer stack
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.extend([
                ConvS2SAttention(hidden_dim, embed_dim, hidden_dim),
                ConvS2SBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    kernel_size=kernel_size,
                    layer_idx=idx,
                    num_layers=num_layers,
                    dropout=dropout
                )
            ])
            
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        output_length: Optional[int] = None
    ) -> torch.Tensor:
        # Length validation
        output_length = output_length or prev_output_tokens.size(1)
        if prev_output_tokens.size(1) > output_length:
            prev_output_tokens = prev_output_tokens[:, :output_length]
            
        # Embeddings
        positions = self.position_ids[:, :prev_output_tokens.size(1)]
        x = self.embed_tokens(prev_output_tokens) + self.embed_positions(positions)
        
        # Layer processing
        for i in range(0, len(self.layers), 2):
            # Attention
            attn_out, _ = self.layers[i](x, encoder_out, encoder_padding_mask)
            x = (x + attn_out)[:, :output_length, :]
            
            # Convolution
            conv_out = self.layers[i+1](x)
            x = conv_out[:, :output_length, :]
            
        return self.proj(x)
        
class ConvS2S(nn.Module):
    """Full model with sequence length validation"""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        target_seq_len: int = 64
    ):
        super().__init__()
        self.encoder = ConvS2SEncoder(
            vocab_size, embed_dim, hidden_dim, num_layers, 
            kernel_size, dropout, max_seq_len
        )
        self.decoder = ConvS2SDecoder(
            vocab_size, embed_dim, hidden_dim, num_layers,
            kernel_size, dropout, max_seq_len
        )
        self.target_seq_len = target_seq_len

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        # Encoder
        src = src[:, -self.encoder.max_positions:]
        encoder_out = self.encoder(src)
        
        # Decoder
        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            tgt = tgt[:, :self.target_seq_len]
            output = self.decoder(
                prev_output_tokens=tgt,
                encoder_out=encoder_out,
                output_length=self.target_seq_len
            )
        else:
            output = self._generate(encoder_out)
            
        return output[:, :self.target_seq_len, :]  # Final length enforcement

    def _generate(self, encoder_out: torch.Tensor) -> torch.Tensor:
        """Autoregressive generation with length control"""
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        outputs = []
        
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for step in range(self.target_seq_len):
            output = self.decoder(
                prev_output_tokens=decoder_input,
                encoder_out=encoder_out,
                output_length=step+1
            )
            outputs.append(output[:, -1:, :])
            decoder_input = torch.cat([
                decoder_input, 
                output[:, -1:].argmax(dim=-1)
            ], dim=1)
            
        return torch.cat(outputs, dim=1)
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, mode: str = 'fixed', 
                 fixed_scale: float = 1.0, learned_scale: float = 1.0):
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

class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        target_seq_len: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_seq_len: int = 512,
        pe_mode: str = 'fixed',
        fixed_scale: float = 1.0,
        learned_scale: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.target_seq_len = target_seq_len
        self.max_seq_len = max_seq_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            mode=pe_mode,
            fixed_scale=fixed_scale,
            learned_scale=learned_scale
        )
        
        # Transformer core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # Final projection layer
        self.fc = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        # Truncate source sequence if needed (keeping right side)
        if src.size(1) > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
            
        # Create source mask for padding tokens
        src_key_padding_mask = (src == 0).to(src.device)
        
        # Embed and add positional encoding to source
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # For training with teacher forcing
        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Prepare target sequence
            if tgt.size(1) > self.target_seq_len:
                tgt = tgt[:, :self.target_seq_len]
            elif tgt.size(1) < self.target_seq_len:
                tgt = torch.nn.functional.pad(tgt, (0, self.target_seq_len - tgt.size(1)), value=0)
            
            # Create target masks
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt_key_padding_mask = (tgt == 0).to(tgt.device)
            
            # Embed and add positional encoding to target
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Transformer forward pass
            out = self.transformer(
                src=src_emb,
                tgt=tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
        # For inference or when not using teacher forcing
        else:
            batch_size = src.size(0)
            device = src.device
            
            # Initialize decoder input with SOS token (assumed to be 1)
            decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            outputs = []
            
            for t in range(self.target_seq_len):
                # Create target mask
                tgt_mask = self.transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                tgt_key_padding_mask = (decoder_input == 0).to(device)
                
                # Embed and add positional encoding
                tgt_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)
                
                # Transformer forward pass
                out = self.transformer(
                    src=src_emb,
                    tgt=tgt_emb,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                
                # Get next token prediction
                next_token = self.fc(out[:, -1:])  # Only take last position
                outputs.append(next_token)
                
                # Update decoder input
                if not self.training:
                    decoder_input = torch.cat([
                        decoder_input,
                        next_token.argmax(dim=-1)
                    ], dim=1)
            
            # Combine all outputs
            out = torch.cat(outputs, dim=1)
        
        # Project to vocabulary size
        return self.fc(out)
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

class Attention(nn.Module):
    """Bahdanau attention mechanism"""
    def __init__(self, hidden_size: int, attention_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size or hidden_size
        
        # Attention layers
        self.attention_hidden = nn.Linear(hidden_size, self.attention_size, bias=False)
        self.attention_context = nn.Linear(hidden_size, self.attention_size, bias=False)
        self.attention_vector = nn.Linear(self.attention_size, 1, bias=False)
        
    def forward(
        self, 
        hidden: torch.Tensor,      # [batch_size, hidden_size]
        encoder_outputs: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        mask: Optional[torch.Tensor] = None  # [batch_size, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and weighted context vector.
        Returns:
            context: Weighted sum of encoder outputs
            attention_weights: Attention distribution over encoder outputs
        """
        seq_len = encoder_outputs.size(1)
        
        # Prepare hidden state
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Calculate attention scores
        attention_hidden = self.attention_hidden(hidden_expanded)  # [batch_size, seq_len, attention_size]
        attention_context = self.attention_context(encoder_outputs)  # [batch_size, seq_len, attention_size]
        attention_sum = torch.tanh(attention_hidden + attention_context)  # [batch_size, seq_len, attention_size]
        attention_scores = self.attention_vector(attention_sum).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Get attention weights and context vector
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            encoder_outputs  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]
        
        return context, attention_weights

class AttentionGRUEncoder(nn.Module):
    """GRU encoder with embeddings"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self, 
        src: torch.Tensor,  # [batch_size, seq_len]
        src_lengths: Optional[torch.Tensor] = None  # [batch_size]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            outputs: Encoder outputs
            hidden: Final hidden state
        """
        # Embed input
        embedded = self.embedding(src)  # [batch_size, seq_len, embed_size]
        
        # Pack sequence if lengths provided
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Pass through GRU
        outputs, hidden = self.gru(embedded)  # outputs: [batch_size, seq_len, hidden_size*num_directions]
        
        # Unpack sequence if packed
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            
        # Combine directions if bidirectional
        if self.gru.bidirectional:
            # Combine directions in outputs
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1)
            outputs = outputs.sum(dim=2)  # [batch_size, seq_len, hidden_size]
            
            # Combine directions in hidden state
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1)
            hidden = hidden.sum(dim=1)  # [num_layers, batch_size, hidden_size]
            
        return outputs, hidden

class AttentionGRUDecoder(nn.Module):
    """GRU decoder with attention mechanism"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(
            input_size=embed_size + hidden_size,  # Concatenate embedding and context
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out = nn.Linear(hidden_size * 2, vocab_size)  # Use both hidden state and context
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        input_step: torch.Tensor,  # [batch_size, 1]
        last_hidden: torch.Tensor,  # [num_layers, batch_size, hidden_size]
        encoder_outputs: torch.Tensor,  # [batch_size, src_seq_len, hidden_size]
        src_mask: Optional[torch.Tensor] = None  # [batch_size, src_seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoder step with attention.
        Returns:
            output: Vocabulary distribution
            hidden: New hidden state
            attention_weights: Attention weights for visualization
        """
        # Get batch size and last hidden state
        batch_size = input_step.size(0)
        last_hidden_top = last_hidden[-1]  # Use top layer hidden state for attention
        
        # Embed input
        embedded = self.embedding(input_step)  # [batch_size, 1, embed_size]
        
        # Calculate attention
        context, attention_weights = self.attention(
            last_hidden_top,
            encoder_outputs,
            src_mask
        )  # context: [batch_size, hidden_size]
        
        # Combine embedding and context
        gru_input = torch.cat([
            embedded,
            context.unsqueeze(1)  # Add sequence dimension
        ], dim=2)  # [batch_size, 1, embed_size + hidden_size]
        
        # GRU step
        output, hidden = self.gru(gru_input, last_hidden)  # output: [batch_size, 1, hidden_size]
        
        # Calculate vocabulary distribution
        output_combined = torch.cat([
            output.squeeze(1),  # Remove sequence dimension
            context
        ], dim=1)  # [batch_size, hidden_size * 2]
        
        # Project to vocabulary size
        output = self.out(output_combined)  # [batch_size, vocab_size]
        
        return output, hidden, attention_weights

class AttentionGRUModel(nn.Module):
    """Complete encoder-decoder model with attention"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        target_seq_len: int = 64,
        max_seq_len: int = 512,
        bidirectional_encoder: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.target_seq_len = target_seq_len
        self.max_seq_len = max_seq_len
        
        # Initialize encoder and decoder
        self.encoder = AttentionGRUEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder
        )
        
        self.decoder = AttentionGRUDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence (optional) [batch_size, tgt_seq_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Sequence of vocabulary distributions [batch_size, tgt_seq_len, vocab_size]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Truncate source sequence if needed
        if src.size(1) > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
            
        # Create source mask (1 for non-pad tokens)
        src_mask = (src != 0).bool()
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # Initialize decoder input with SOS token (assumed to be 1)
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Initialize decoder hidden state with encoder's final hidden state
        decoder_hidden = encoder_hidden
        
        # Prepare storage for decoder outputs
        outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)
        
        # Generate target sequence
        for t in range(self.target_seq_len):
            # Decoder forward pass
            output, decoder_hidden, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                src_mask
            )
            
            # Store output
            outputs[:, t] = output
            
            # Determine next input
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1]  # Teacher forcing
            else:
                decoder_input = output.argmax(dim=1, keepdim=True)  # Use own prediction
                
        return outputs
