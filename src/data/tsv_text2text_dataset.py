# src/data/tsv_text2text_dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
import itertools
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

@dataclass
class ColumnConfig:
    """Configuration for column processing"""
    file_path: str
    source_columns: List[Union[str, int]]
    target_columns: List[Union[str, int]]
    has_header: bool = True
    separator: str = "\t"
    join_token: str = " "

@dataclass
class CacheConfig:
    """Configuration for dataset caching"""
    enable_cache: bool = True
    cache_dir: str = "./cache"
    cache_format: str = "h5"  # 'h5' or 'mmap'
    preload_cache: bool = False  # Whether to load entire cache into memory

class CachedTSVDataset(Dataset):
    def __init__(
        self,
        configs: List[ColumnConfig],
        cache_config: CacheConfig,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        max_length: int = 512,
        seed: int = 42
    ):
        self.configs = configs
        self.cache_config = cache_config
        self.max_length = max_length
        
        # Setup caching
        self.cache_dir = Path(cache_config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = self._setup_tokenizer(tokenizer_path, vocab_size)
        
        # Create or load cache
        self.cache_path = self._get_cache_path()
        self.data_cache = self._setup_cache()
        
        # Initialize indices
        self._setup_indices()

    def _read_chunks(self, config: ColumnConfig):
        """Read data in chunks"""
        try:
            # Create column names mapping
            source_cols = [f'col_{i}' for i in config.source_columns]
            target_cols = [f'col_{i}' for i in config.target_columns]
            all_cols = source_cols + target_cols
            col_mapping = dict(zip(range(len(all_cols)), all_cols))
            
            for chunk in pd.read_csv(
                config.file_path,
                sep=config.separator,
                usecols=config.source_columns + config.target_columns,
                header=None,  # Always treat as no header
                names=all_cols,  # Use our generated column names
                chunksize=10000,
                dtype=str,
                on_bad_lines='skip'
            ):
                # Process source and target text
                source_text = chunk[source_cols].astype(str).agg(config.join_token.join, axis=1)
                target_text = chunk[target_cols].astype(str).agg(config.join_token.join, axis=1)
                
                # Create DataFrame with processed text
                processed_chunk = pd.DataFrame({
                    'source': source_text.values,
                    'target': target_text.values
                })
                
                yield processed_chunk
                
        except Exception as e:
            logging.error(f"Error reading file {config.file_path}: {str(e)}")
            yield pd.DataFrame(columns=['source', 'target'])

    def _get_cache_path(self) -> Path:
        """Generate unique cache path based on dataset configuration"""
        config_str = str(sorted([
            (c.file_path, c.source_columns, c.target_columns) 
            for c in self.configs
        ]))
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"dataset_cache_{cache_hash}.{self.cache_config.cache_format}"

    def _setup_cache(self) -> Union[h5py.File, np.memmap]:
        """Setup cache file based on configuration"""
        if self.cache_path.exists():
            if self.cache_config.cache_format == 'h5':
                return h5py.File(self.cache_path, 'r')
            else:  # mmap
                return np.load(self.cache_path, mmap_mode='r')
        else:
            return self._create_cache()

    def _create_cache(self) -> Union[h5py.File, np.memmap]:
        """Create and populate cache file"""
        logging.info(f"Creating cache file at {self.cache_path}")
        
        # Process all data first to get dimensions
        all_source_ids = []
        all_target_ids = []
        
        for config in self.configs:
            for chunk in self._read_chunks(config):
                source_encodings = self.tokenizer.encode_batch(chunk['source'].tolist())
                target_encodings = self.tokenizer.encode_batch(chunk['target'].tolist())
                
                for src, tgt in zip(source_encodings, target_encodings):
                    all_source_ids.append(src.ids[:self.max_length])
                    all_target_ids.append(tgt.ids[:self.max_length])
        
        # Create cache file
        if self.cache_config.cache_format == 'h5':
            with h5py.File(self.cache_path, 'w') as f:
                f.create_dataset('source_ids', data=np.array(all_source_ids, dtype=np.int32))
                f.create_dataset('target_ids', data=np.array(all_target_ids, dtype=np.int32))
                f.create_dataset('lengths', data=np.array(
                    [(len(src), len(tgt)) for src, tgt in zip(all_source_ids, all_target_ids)],
                    dtype=np.int32
                ))
            return h5py.File(self.cache_path, 'r')
        
        else:  # mmap
            data = np.array(list(zip(all_source_ids, all_target_ids)), dtype=np.int32)
            np.save(self.cache_path, data)
            return np.load(self.cache_path, mmap_mode='r')

    def _setup_indices(self):
        """Setup indices for dataset access"""
        if self.cache_config.cache_format == 'h5':
            self.length = len(self.data_cache['source_ids'])
        else:
            self.length = len(self.data_cache)

    def _setup_tokenizer(self, tokenizer_path: Optional[str], vocab_size: int) -> Tokenizer:
        """Initialize or load tokenizer"""
        if tokenizer_path and Path(tokenizer_path).exists():
            logging.info(f"Loading tokenizer from {tokenizer_path}")
            return Tokenizer.from_file(tokenizer_path)
        
        logging.info("Creating and training new tokenizer")
        # Create new tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2
        )
        
        # Train tokenizer on dataset
        def text_iterator():
            all_texts = []
            for config in self.configs:
                for chunk in self._read_chunks(config):
                    if not chunk.empty:
                        # Combine source and target texts
                        texts = chunk['source'].tolist() + chunk['target'].tolist()
                        # Filter out None or empty strings
                        texts = [str(text) for text in texts if text is not None and str(text).strip()]
                        all_texts.extend(texts)
            return all_texts
        
        # Train the tokenizer
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        
        # Save tokenizer
        if self.cache_dir:
            tokenizer_save_path = self.cache_dir / "tokenizer.json"
            tokenizer.save(str(tokenizer_save_path))
            logging.info(f"Saved tokenizer to {tokenizer_save_path}")
        
        return tokenizer

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer"""
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_config.cache_format == 'h5':
            source_ids = self.data_cache['source_ids'][idx]
            target_ids = self.data_cache['target_ids'][idx]
        else:
            source_ids, target_ids = self.data_cache[idx]

        return {
            'source_text': torch.tensor(source_ids, dtype=torch.long),
            'target_text': torch.tensor(target_ids, dtype=torch.long)
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
