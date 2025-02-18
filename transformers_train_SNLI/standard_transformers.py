
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import time


# Toy dataset for text summarization (input: random tokens, target: first 3 tokens)
class ToyDataset(Dataset):
    def __init__(self, vocab_size=100, seq_len=10, num_samples=1000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = x[:3]  # Summarize by taking first 3 tokens
        return x, y

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
            pe = torch.zeros(1, max_len, d_model)  # Added batch dimension
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
            
        seq_len = x.size(1)  # Changed from 0 to 1 for batch_first
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
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_seq_len: int = 5000,
                 pe_mode: str = 'fixed', fixed_scale: float = 1.0,
                 learned_scale: float = 1.0):
        
        super().__init__()
        self.d_model = d_model

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
        
        # Transformer core (now with batch_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Changed to True
        )
        
        # Final projection layer
        self.fc = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Create masks
        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = None
        tgt_padding_mask = None

        # Convert to embeddings (batch_first is maintained throughout)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encodings
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Pass through transformer
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None
        )

        # Project to vocabulary size
        return self.fc(out)

# Training setup
def train_transformer():
    vocab_size = 100
    model = VanillaTransformer(
        vocab_size=vocab_size, 
        d_model=256, 
        nhead=4, 
        pe_mode='fixed'
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    model = model.to(device)
    
    dataset = ToyDataset(vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        total_loss = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Create input and target sequences for teacher forcing
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]   # Remove first token
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    start_time = time.time()  # Inicia el temporizador
    train_transformer()
    end_time = time.time()  # Termina el temporizador
    
    elapsed_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos")