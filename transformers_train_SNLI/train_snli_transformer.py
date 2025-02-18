import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
import math
import time
import json

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
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_seq_len: int = 5000,
                 pe_mode: str = 'fixed', fixed_scale: float = 1.0,
                 learned_scale: float = 1.0):
        
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            mode=pe_mode,
            fixed_scale=fixed_scale,
            learned_scale=learned_scale
        )
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
        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = None
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = None
        tgt_padding_mask = None

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

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

        return self.fc(out)

class DataLoader_custom(Dataset):
    def __init__(self, data_dir, split):
        with open(data_dir, 'r') as file:
            dic_data = json.load(file)
        self.X, self.y = self.separar(dic_data, split)
        self.samples = len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.samples
    
    def separar(self, data, split):
        X=[]
        y=[]
        for i in range(len(data[split])):
            X.append(self.tokenize(data[split][i][0]))
            y.append(self.tokenize(data[split][i][1]))
        return X, y

    def tokenize(self, text):
        # Tokenize the text by splitting on spaces and converting to indices
        # Here, we use a simple approach, but you may want to use a more sophisticated tokenizer
        return [ord(char) for char in text]

def collate_fn(batch):
    X, y = zip(*batch)
    X = [torch.tensor(x) for x in X]
    y = [torch.tensor(y) for y in y]
    return torch.nn.utils.rnn.pad_sequence(X, batch_first=True), torch.nn.utils.rnn.pad_sequence(y, batch_first=True)

def train_transformer():
    vocab_size = 256  # Maximum value of ord(char) is 255
    model = VanillaTransformer(
        vocab_size=vocab_size, 
        d_model=256, 
        nhead=4, 
        pe_mode='fixed'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    model = model.to(device)
    
    data_dir = "SNLI_procesados_tuplas.json"
    split = "dev"
    dataset = DataLoader_custom(data_dir, split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(7):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/10], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/7], Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    train_transformer()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos")
