# src/models/text2text_autoencoders.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple
import numpy as np

class WeightNormConv1d(nn.Module):
    """Weight-normalized 1D convolution with proper causal padding"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0  # No symmetric padding
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.conv.bias, 0.0)
        with torch.no_grad():
            weight_flat = self.conv.weight.view(out_channels, -1)
            weight_norms = torch.norm(weight_flat, dim=1, p=2)
            self.scale = nn.Parameter(weight_norms.clone())
            self.conv.weight.data = F.normalize(weight_flat, dim=1).view_as(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_flat = self.conv.weight.view(self.conv.out_channels, -1)
        weight = self.scale.view(-1, 1, 1) * F.normalize(weight_flat, dim=1).view_as(self.conv.weight)
        pad_left = self.conv.kernel_size[0] - 1
        x = F.pad(x, (pad_left, 0))
        return F.conv1d(x, weight, self.conv.bias, padding=0)

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
        self.conv = WeightNormConv1d(
            in_channels=input_dim,
            out_channels=2 * output_dim,
            kernel_size=kernel_size
        )
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = F.glu(x, dim=-1)
        if self.residual is not None:
            residual = self.residual(residual)
        x = (x + residual) / math.sqrt(self.num_layers)
        x = self.layer_norm(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return self.dropout(x)

class ConvS2SAttention(nn.Module):
    def __init__(self, decoder_dim: int, encoder_dim: int, hidden_dim: int):
        super().__init__()
        self.decoder_proj = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.context_proj = nn.Linear(encoder_dim, decoder_dim, bias=False)  # New projection
        self.scaling = 1.0 / math.sqrt(hidden_dim)
        
    def forward(self, decoder_state: torch.Tensor, encoder_out: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        decoder_hidden = self.decoder_proj(decoder_state)
        encoder_hidden = self.encoder_proj(encoder_out)
        decoder_hidden = decoder_hidden.unsqueeze(2)
        encoder_hidden = encoder_hidden.unsqueeze(1)
        combined = torch.tanh(decoder_hidden + encoder_hidden) * self.scaling
        attn_scores = self.output_proj(combined).squeeze(-1)
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(encoder_padding_mask.unsqueeze(1), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_out)  # (batch_size, tgt_len, encoder_dim)
        context = self.context_proj(context)  # (batch_size, tgt_len, decoder_dim)
        return context, attn_weights

class ConvS2SEncoder(nn.Module):
    """Convolutional encoder with improved sequence handling"""
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
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_dim ** -0.5)
        self.embed_positions = nn.Embedding(max_positions, embed_dim)
        nn.init.normal_(self.embed_positions.weight, mean=0, std=embed_dim ** -0.5)
        positions = torch.arange(max_positions).unsqueeze(0)
        self.register_buffer('positions', positions)
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
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
        self.output_projection = nn.Linear(hidden_dim, embed_dim) if hidden_dim != embed_dim else None
        
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if src_tokens.size(1) > self.max_positions:
            src_tokens = src_tokens[:, :self.max_positions]
            if src_mask is not None:
                src_mask = src_mask[:, :self.max_positions]
        positions = self.positions[:, :src_tokens.size(1)]
        x = self.embed_tokens(src_tokens) + self.embed_positions(positions)
        x = self.embed_layer_norm(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        if self.output_projection is not None:
            x = self.output_projection(x)
        return x

class ConvS2SDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 4, kernel_size: int = 3, dropout: float = 0.1, max_positions: int = 512, use_attention: bool = True):
        super().__init__()
        self.max_positions = max_positions
        self.use_attention = use_attention
        
        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=embed_dim ** -0.5)
        self.embed_positions = nn.Embedding(max_positions, embed_dim)
        nn.init.normal_(self.embed_positions.weight, mean=0, std=embed_dim ** -0.5)
        self.register_buffer('position_ids', torch.arange(max_positions).unsqueeze(0))
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        
        # Project embed_dim to hidden_dim if they differ
        if embed_dim != hidden_dim:
            self.input_proj = nn.Linear(embed_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()  # Use Identity to avoid None checks
        
        # Decoder layers
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if use_attention:
                self.layers.append(ConvS2SAttention(hidden_dim, embed_dim, hidden_dim))
            self.layers.append(ConvS2SBlock(
                input_dim=hidden_dim,  # Input is always hidden_dim after projection
                output_dim=hidden_dim,
                kernel_size=kernel_size,
                layer_idx=idx,
                num_layers=num_layers,
                dropout=dropout
            ))
        
        # Output projection
        self.proj = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.proj.weight, mean=0, std=hidden_dim ** -0.5)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, prev_output_tokens: torch.Tensor, encoder_out: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None, output_length: Optional[int] = None) -> torch.Tensor:
        seq_len = min(prev_output_tokens.size(1), output_length or self.max_positions)
        prev_output_tokens = prev_output_tokens[:, :seq_len]
        
        positions = self.position_ids[:, :prev_output_tokens.size(1)]
        x = self.embed_tokens(prev_output_tokens) + self.embed_positions(positions)
        x = self.embed_layer_norm(x)
        
        # Apply projection (Identity if embed_dim == hidden_dim)
        x = self.input_proj(x)
        
        # Process through layers
        for layer in self.layers:
            if self.use_attention and isinstance(layer, ConvS2SAttention):
                attn_out, _ = layer(x, encoder_out, encoder_padding_mask)
                x = x + attn_out
            else:
                x = layer(x)
        
        return self.proj(x)

class ConvS2S(nn.Module):
    """
    Complete Convolutional Sequence-to-Sequence model with optional attention.
    Updated to use source_seq_len instead of max_seq_len, so no extra ambiguity.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        source_seq_len: int = 512,  # replaces max_seq_len
        target_seq_len: int = 64,
        use_attention: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.pad_id = 0
        self.sos_id = None
        self.eos_id = None

        # Encoder and Decoder
        self.encoder = ConvS2SEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            max_positions=self.source_seq_len
        )
        self.decoder = ConvS2SDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            max_positions=self.target_seq_len,
            use_attention=use_attention
        )

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Truncate source if needed
        if src.size(1) > self.source_seq_len:
            src = src[:, -self.source_seq_len:]

        # Build a mask for padding tokens (pad_idx=0 or custom if needed)
        src_padding_mask = (src == self.pad_id)
        encoder_out = self.encoder(src, src_padding_mask)
        
        # Decide training vs generation
        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # ensure target is sized up to self.target_seq_len
            if tgt.size(1) > self.target_seq_len:
                tgt = tgt[:, :self.target_seq_len]
            elif tgt.size(1) < self.target_seq_len:
                pad_len = self.target_seq_len - tgt.size(1)
                tgt = torch.nn.functional.pad(tgt, (0, pad_len), value=self.pad_id)

            decoder_out = self.decoder(
                prev_output_tokens=tgt,
                encoder_out=encoder_out,
                encoder_padding_mask=src_padding_mask,
                output_length=self.target_seq_len
            )
        else:
            decoder_out = self._generate(encoder_out, src_padding_mask)

        return decoder_out

    def _generate(self, encoder_out: torch.Tensor, encoder_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autoregressive generation loop."""
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # SOS token assumed to be 1 by default, or customize if needed
        decoder_input = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)
        outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)

        for step in range(self.target_seq_len):
            out = self.decoder(
                prev_output_tokens=decoder_input,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                output_length=step + 1
            )
            outputs[:, step:step+1, :] = out[:, -1:, :]
            next_token = out[:, -1:, :].argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        return outputs

    
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

class LSTMAttention(nn.Module):
    """Bahdanau-style attention mechanism for LSTM decoder"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # Combine decoder hidden and encoder output
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch_size, hidden_size] - Decoder's current hidden state
            encoder_outputs: [batch_size, src_seq_len, hidden_size] - Encoder's output
        Returns:
            context: [batch_size, hidden_size] - Context vector
            attn_weights: [batch_size, src_seq_len] - Attention weights
        """
        batch_size, src_seq_len, _ = encoder_outputs.size()
        
        hidden = hidden.unsqueeze(1).repeat(1, src_seq_len, 1)  # [batch_size, src_seq_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_seq_len, hidden_size]
        energy = energy.transpose(1, 2)  # [batch_size, hidden_size, src_seq_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hidden_size]
        attn_scores = torch.bmm(v, energy).squeeze(1)  # [batch_size, src_seq_len]
        
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, src_seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, hidden_size]
        
        return context, attn_weights

class AttentionLSTMSeq2Seq(nn.Module):
    """
    Bahdanau-style LSTM with optional attention for seq2seq.
    Updated to rename max_seq_len -> source_seq_len.
    """
    def __init__(
        self,
        vocab_size: int,
        source_seq_len: int = 512,  # was max_seq_len
        target_seq_len: int = 64,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional_encoder: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.use_attention = use_attention
        self.dropout = dropout

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM encoder
        encoder_hidden_size = hidden_size // 2 if bidirectional_encoder else hidden_size
        self.encoder_lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional_encoder
        )

        # Decoder
        decoder_input_size = embed_size + (encoder_hidden_size * 2 if bidirectional_encoder else encoder_hidden_size) if use_attention else embed_size
        self.decoder_lstm = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = LSTMAttention(hidden_size) if use_attention else None

        # Output projection
        fc_input_size = hidden_size + (encoder_hidden_size * 2 if bidirectional_encoder else encoder_hidden_size) if use_attention else hidden_size
        self.fc = nn.Linear(fc_input_size, vocab_size)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Weight init
        self._init_weights()
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Truncate source if needed
        if src.size(1) > self.source_seq_len:
            src = src[:, -self.source_seq_len:]

        batch_size, device = src.size(0), src.device

        # Encode
        src_emb = self.dropout_layer(self.embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(src_emb)

        # Adjust if bidirectional
        if self.bidirectional_encoder:
            hidden = hidden.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)
            cell = cell.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)

        # Training with teacher forcing
        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            if tgt.size(1) > self.target_seq_len:
                tgt = tgt[:, :self.target_seq_len]
            else:
                pad_len = self.target_seq_len - tgt.size(1)
                if pad_len > 0:
                    tgt = torch.nn.functional.pad(tgt, (0, pad_len), value=0)

            tgt_emb = self.dropout_layer(self.embedding(tgt))
            outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)
            decoder_input = tgt_emb[:, 0, :]

            for t in range(self.target_seq_len):
                if self.use_attention:
                    context, _ = self.attention(hidden[-1], encoder_outputs)
                    decoder_input_with_context = torch.cat((decoder_input, context), dim=-1)
                else:
                    decoder_input_with_context = decoder_input

                decoder_output, (hidden, cell) = self.decoder_lstm(
                    decoder_input_with_context.unsqueeze(1), (hidden, cell)
                )
                if self.use_attention:
                    combined_output = torch.cat((decoder_output.squeeze(1), context), dim=-1)
                else:
                    combined_output = decoder_output.squeeze(1)

                step_output = self.fc(combined_output)
                outputs[:, t:t+1, :] = step_output.unsqueeze(1)

                # Update decoder_input
                if t < self.target_seq_len - 1:
                    decoder_input = tgt_emb[:, t+1, :]
                else:
                    decoder_input = torch.zeros_like(decoder_input)

            return outputs
        
        # Inference
        else:
            outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)
            decoder_input = self.embedding(torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)).squeeze(1)

            for t in range(self.target_seq_len):
                if self.use_attention:
                    context, _ = self.attention(hidden[-1], encoder_outputs)
                    decoder_input_with_context = torch.cat((decoder_input, context), dim=-1)
                else:
                    decoder_input_with_context = decoder_input

                decoder_output, (hidden, cell) = self.decoder_lstm(
                    decoder_input_with_context.unsqueeze(1), (hidden, cell)
                )
                if self.use_attention:
                    combined_output = torch.cat((decoder_output.squeeze(1), context), dim=-1)
                else:
                    combined_output = decoder_output.squeeze(1)

                step_output = self.fc(combined_output)
                outputs[:, t] = step_output

                # Next input
                next_token = step_output.argmax(dim=-1)
                decoder_input = self.embedding(next_token)

            return outputs

# src/models/text2text_autoencoders.py

class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        target_seq_len: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        # Remove independent max_seq_len parameter; use source_seq_len instead.
        source_seq_len: int = 64,
        pe_mode: str = 'fixed',
        fixed_scale: float = 1.0,
        learned_scale: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.target_seq_len = target_seq_len
        self.source_seq_len = source_seq_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Initialize positional encoder with a maximum length that covers both source and
        # potential decoder sequence lengths (decoder may temporarily grow to target_seq_len+1).
        max_len_for_pos = max(self.source_seq_len, self.target_seq_len + 1)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len_for_pos,
            mode=pe_mode,
            fixed_scale=fixed_scale,
            learned_scale=learned_scale
        )
        
        # Transformer core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers if num_decoder_layers is None else num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        # Final projection layer
        self.fc = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate a causal mask for size sz:
        - 0.0 in the lower-triangular part (allowing tokens to attend to earlier tokens)
        - -inf in the upper-triangular part (preventing tokens from attending to future tokens)
        """
        # Create a matrix of shape [sz, sz] filled with float(0.0)
        mask = torch.zeros(sz, sz)
        # Fill everything above the main diagonal with -inf
        mask = mask.fill_(float(0.0)).float()
        mask = mask.masked_fill(torch.triu(torch.ones(sz, sz), diagonal=1).bool(), float('-inf'))
        return mask


    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Truncate source sequence if needed (keeping right side)
        if src.size(1) > self.source_seq_len:
            src = src[:, -self.source_seq_len:]
        
        # Create source mask for padding tokens
        src_key_padding_mask = (src == self.pad_id).to(src.device)

        # Embed and add positional encoding to source
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # For training with teacher forcing
        if self.training and tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
            # Prepare target sequence using target_seq_len
            if tgt.size(1) > self.target_seq_len:
                tgt = tgt[:, :self.target_seq_len]
            elif tgt.size(1) < self.target_seq_len:
                tgt = torch.nn.functional.pad(tgt, (0, self.target_seq_len - tgt.size(1)), value=0)
            
            # Create target masks
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            tgt_key_padding_mask = (tgt == self.pad_id).to(tgt.device)

            
            # Embed and add positional encoding to target
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Transformer forward pass with teacher forcing
            out = self.transformer(
                src=src_emb,
                tgt=tgt_emb,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            result = self.fc(out)
            # Clean up intermediates to free memory
            del out, tgt_emb, tgt_mask, tgt_key_padding_mask
            
            return result
        
        # For inference or when not using teacher forcing
        else:
            batch_size = src.size(0)
            device = src.device
            
            # Initialize decoder input with SOS token (assumed to be 1)
            decoder_input = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)
            outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)  # Pre-allocate outputs
            
            for t in range(self.target_seq_len):
                # Create target mask
                tgt_mask = self.transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
                tgt_key_padding_mask = (decoder_input == self.pad_id).to(device)
                
                # Embed and add positional encoding to decoder input
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
                outputs[:, t:t+1, :] = next_token  # Fill pre-allocated tensor
                
                # Update decoder input (only in inference mode)
                if not self.training:
                    decoder_input = torch.cat([decoder_input, next_token.argmax(dim=-1)], dim=1)
            # Final cleanup
            del src_emb, tgt_emb, tgt_mask, tgt_key_padding_mask, decoder_input
        
            return outputs

        
class PositionalAutoencoder(nn.Module):
    """
    Simple (non-autoregressive) autoencoder that uses positional encodings.
    Updated to rename max_seq_len -> source_seq_len, removing ambiguity.
    """
    def __init__(
        self,
        vocab_size: int,
        target_seq_len: int = 64,
        d_model: int = 512,
        hidden_dim: int = 256,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        source_seq_len: int = 5000,    # replaces old max_seq_len
        pe_mode: str = 'fixed',
        use_normalization: bool = True,
        norm_type: str = 'batch',
        fixed_scale: float = 1.0,
        learned_scale: float = 1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.source_seq_len = source_seq_len  # was max_seq_len
        self.target_seq_len = target_seq_len
        self.use_normalization = use_normalization
        self.norm_type = norm_type

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=source_seq_len,  # pass in new param
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

        # Bottleneck
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

        # Final projection
        self.fc = nn.Linear(d_model, vocab_size)
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, 
                tgt: Optional[torch.Tensor] = None,              # IGNORE this
                teacher_forcing_ratio: float = 1.0               # IGNORE this
               ) -> torch.Tensor:
        """
        Non-autoregressive. Output size = (batch_size, target_seq_len, vocab_size)
        """
        if len(src.shape) != 2:
            raise ValueError(f"Expected 2D input tensor (batch_size, seq_len), got shape {src.shape}")

        batch_size, src_len = src.size(0), src.size(1)
        if src_len > self.source_seq_len:
            # truncate right side
            src = src[:, -self.source_seq_len:]

        # embed + positional encode
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)  # shape = (batch_size, seq_len, d_model)

        # Flatten for encoder
        x = x.reshape(batch_size * x.size(1), self.d_model)
        x = self.encoder(x)  # (batch_size * seq_len, hidden_dim)

        # Reshape + average across sequence
        x = x.reshape(batch_size, -1, self.hidden_dim)
        x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # Bottleneck
        x = self.bottleneck(x)  # (batch_size, bottleneck_dim * target_seq_len)
        x = x.reshape(batch_size, self.target_seq_len, self.bottleneck_dim)

        # Decoder
        x = x.reshape(batch_size * self.target_seq_len, self.bottleneck_dim)
        x = self.decoder(x)  # (batch_size * target_seq_len, d_model)
        x = x.reshape(batch_size, self.target_seq_len, self.d_model)

        # Final projection
        logits = self.fc(x)
        return logits


class GRUAttention(nn.Module):
    """Bahdanau attention mechanism"""
    def __init__(self, hidden_size: int, attention_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size or hidden_size
        
        self.attention_hidden = nn.Linear(hidden_size, self.attention_size, bias=False)
        self.attention_context = nn.Linear(hidden_size, self.attention_size, bias=False)
        self.attention_vector = nn.Linear(self.attention_size, 1, bias=False)
        
    def forward(
        self, 
        hidden: torch.Tensor,      # [batch_size, hidden_size]
        encoder_outputs: torch.Tensor,  # [batch_size, seq_len, hidden_size]
        mask: Optional[torch.Tensor] = None  # [batch_size, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = encoder_outputs.size(1)
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)
        attention_hidden = self.attention_hidden(hidden_expanded)
        attention_context = self.attention_context(encoder_outputs)
        attention_sum = torch.tanh(attention_hidden + attention_context)
        attention_scores = self.attention_vector(attention_sum).squeeze(-1)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
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
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self._init_weights()
        
    def _init_weights(self):
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
        embedded = self.embedding(src)
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        outputs, hidden = self.gru(embedded)
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if self.gru.bidirectional:
            outputs = outputs.view(outputs.size(0), outputs.size(1), 2, -1).sum(dim=2)
            hidden = hidden.view(self.num_layers, 2, hidden.size(1), -1).sum(dim=1)
        return outputs, hidden

class AttentionGRUDecoder(nn.Module):
    """GRU decoder with optional attention mechanism"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_attention: bool = True  # New parameter to toggle attention
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Attention (only if enabled)
        self.attention = GRUAttention(hidden_size) if use_attention else None
        
        # GRU input size depends on whether attention is used
        gru_input_size = embed_size + hidden_size if use_attention else embed_size
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer size depends on attention
        out_input_size = hidden_size * 2 if use_attention else hidden_size
        self.out = nn.Linear(out_input_size, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
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
        batch_size = input_step.size(0)
        last_hidden_top = last_hidden[-1]
        
        embedded = self.embedding(input_step)  # [batch_size, 1, embed_size]
        
        if self.use_attention:
            # Calculate attention
            context, attention_weights = self.attention(last_hidden_top, encoder_outputs, src_mask)
            gru_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [batch_size, 1, embed_size + hidden_size]
        else:
            # No attention: use embedding directly
            gru_input = embedded
            context = torch.zeros(batch_size, self.hidden_size, device=input_step.device)
            attention_weights = None
        
        output, hidden = self.gru(gru_input, last_hidden)
        
        # Combine GRU output with context (if attention is used)
        output_combined = torch.cat([output.squeeze(1), context], dim=1) if self.use_attention else output.squeeze(1)
        output = self.out(output_combined)
        
        return output, hidden, attention_weights

class AttentionGRUModel(nn.Module):
    """
    Complete encoder-decoder GRU model with optional attention.
    Updated to rename max_seq_len -> source_seq_len for clarity.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        source_seq_len: int = 512,  # replaced old max_seq_len
        target_seq_len: int = 64,
        bidirectional_encoder: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len

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
            dropout=dropout,
            use_attention=use_attention
        )
        self.pad_id = None
        self.sos_id = None
        self.eos_id = None

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        # Truncate source
        if src.size(1) > self.source_seq_len:
            src = src[:, -self.source_seq_len:]

        batch_size = src.size(0)
        device = src.device

        # Build mask (pad_idx=0 assumed)
        src_mask = (src != self.pad_id).bool()
        encoder_outputs, encoder_hidden = self.encoder(src)

        # For generation logic
        decoder_input = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden

        # Pre-allocate outputs
        outputs = torch.zeros(batch_size, self.target_seq_len, self.vocab_size, device=device)

        for t in range(self.target_seq_len):
            output, decoder_hidden, _ = self.decoder(
                input_step=decoder_input,
                last_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask
            )
            outputs[:, t] = output

            if (self.training and tgt is not None 
                    and torch.rand(1).item() < teacher_forcing_ratio 
                    and t < tgt.size(1)):
                # teacher forcing
                decoder_input = tgt[:, t:t+1]
            else:
                # choose next token from model output
                next_token = output.argmax(dim=-1, keepdim=True)
                decoder_input = next_token

        return outputs
