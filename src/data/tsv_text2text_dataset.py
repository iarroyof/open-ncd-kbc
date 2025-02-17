import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json
from dataclasses import dataclass
from torch.utils.data._utils.collate import default_collate
import logging

@dataclass
class ColumnConfig:
    """Configuration for column processing"""
    file_path: str
    source_columns: List[Union[str, int]]  # Can be column names or indices
    target_columns: List[Union[str, int]]  # Can be column names or indices
    has_header: bool = True  # Indicates if file has header row
    separator: str = "\t"
    join_token: str = " "

class TSVText2TextDataset(Dataset):
    def __init__(
        self,
        configs: List[ColumnConfig],
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        chunk_size: int = 10000
    ):
        """
        Initialize the dataset with configurations for each input file.
        
        Args:
            configs: List of ColumnConfig objects defining source/target columns
            tokenizer_path: Path to load pretrained tokenizer (if None, will train new)
            vocab_size: Size of vocabulary for tokenizer training
            max_length: Maximum sequence length
            cache_dir: Directory to cache tokenizer and processed data
            seed: Random seed for shuffling
            chunk_size: Number of rows to process at once

            # Example usage:
            # Example with named columns (files with headers)
            configs_with_headers = [
                ColumnConfig(
                    file_path="data1.tsv",
                    source_columns=["input", "context"],
                    target_columns=["output"],
                    has_header=True
                ),
                ColumnConfig(
                    file_path="data2.tsv",
                    source_columns=["question"],
                    target_columns=["answer"],
                    has_header=True
                )
            ]
            
            # Example with column indices (files without headers)
            configs_without_headers = [
                ColumnConfig(
                    file_path="data3.tsv",
                    source_columns=[0, 1],  # First and second columns as source
                    target_columns=[2],     # Third column as target
                    has_header=False
                ),
                ColumnConfig(
                    file_path="data4.tsv",
                    source_columns=[0],     # First column as source
                    target_columns=[1],     # Second column as target
                    has_header=False
                )
            ]
            
            dataset = TSVText2TextDataset(
                configs=configs,
                cache_dir="./cache",
                seed=42
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4
            )
        """
        self.configs = configs
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize or load tokenizer
        self.tokenizer = self._setup_tokenizer(tokenizer_path, vocab_size)
        
        # Calculate total size and create index mapping
        self.total_size = 0
        self.file_indices = []
        
        for config in configs:
            # Prepare usecols parameter based on whether we're using names or indices
            if config.has_header:
                usecols = [0]  # Just use first column for counting when header exists
            else:
                usecols = [0]  # Use first column index for counting
            
            # Count rows, accounting for header
            file_size = sum(1 for _ in pd.read_csv(
                config.file_path, 
                sep=config.separator,
                usecols=usecols,
                header=0 if config.has_header else None,
                chunksize=chunk_size
            ))
            
            # Adjust file size if header was counted
            if config.has_header:
                file_size -= 1
                
            self.file_indices.extend([(self.total_size + i, config) 
                                    for i in range(file_size)])
            self.total_size += file_size
        
        # Shuffle indices
        np.random.shuffle(self.file_indices)

    def _setup_tokenizer(self, tokenizer_path: Optional[str], vocab_size: int) -> Tokenizer:
        """Initialize or load tokenizer"""
        if tokenizer_path and Path(tokenizer_path).exists():
            return Tokenizer.from_file(tokenizer_path)
        
        # Create new tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        )
        
        # Train tokenizer on dataset
        def text_iterator():
            for config in self.configs:
                for chunk in pd.read_csv(config.file_path, sep=config.separator, chunksize=self.chunk_size):
                    # Combine source and target columns
                    all_text = []
                    for cols in [config.source_columns, config.target_columns]:
                        text = chunk[cols].fillna("").agg(config.join_token.join, axis=1)
                        all_text.extend(text.tolist())
                    yield all_text
        
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        
        # Save tokenizer
        if self.cache_dir:
            tokenizer.save(str(self.cache_dir / "tokenizer.json"))
        
        return tokenizer

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        global_idx, config = self.file_indices[idx]
        
        # Calculate chunk and local index
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size
        
        # Prepare column specifications
        source_cols = config.source_columns
        target_cols = config.target_columns
        
        # Read specific chunk
        chunk = next(itertools.islice(
            pd.read_csv(
                config.file_path,
                sep=config.separator,
                header=0 if config.has_header else None,
                names=None if config.has_header else [f"col_{i}" for i in range(1000)],  # Large enough for any reasonable number of columns
                usecols=source_cols + target_cols if config.has_header else list(set(source_cols + target_cols)),
                chunksize=self.chunk_size
            ),
            chunk_idx, chunk_idx + 1
        ))
        
        # Process source and target text
        source_text = config.join_token.join(
            [str(chunk.iloc[local_idx][col]) for col in config.source_columns]
        )
        target_text = config.join_token.join(
            [str(chunk.iloc[local_idx][col]) for col in config.target_columns]
        )
        
        # Tokenize
        source_encoding = self.tokenizer.encode(source_text)
        target_encoding = self.tokenizer.encode(target_text)
        
        return {
            "source_text": torch.tensor(
                source_encoding.ids[:self.max_length], 
                dtype=torch.long
            ),
            "target_text": torch.tensor(
                target_encoding.ids[:self.max_length],
                dtype=torch.long
            )
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for padding sequences"""
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in batch],
            batch_first=True,
            padding_value=0  # Assuming 0 is the padding token ID
        )
        for key in batch[0].keys()
    }
