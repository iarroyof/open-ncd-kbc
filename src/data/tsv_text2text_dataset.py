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
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
import re

@dataclass
class ColumnConfig:
    """Configuration for column processing"""
    file_path: str
    source_columns: List[Union[str, int]]
    target_columns: List[Union[str, int]]
    has_header: bool = True
    separator: str = "\t"
    join_token: str = " "
    camel_to_lower: Optional[List[int]] = None  # Columns to convert from CamelCase to lower case

@dataclass
class CacheConfig:
    """Configuration for dataset caching"""
    enable_cache: bool = True
    cache_dir: str = "./cache"
    cache_format: str = "h5"
    preload_cache: bool = False
    tokenizer_path: str = "model_tokenizer.json"  # Model-specific tokenizer file

class CachedTSVDataset(Dataset):
    def __init__(
        self,
        configs: List[ColumnConfig],
        cache_config: CacheConfig,
        vocab_size: int = 32000,
        max_length: int = 512,  # Model-specific source_seq_len
        target_length: int = 64,  # Model-specific target_seq_len
        seed: int = 42
    ):
        self.configs = configs
        self.cache_config = cache_config
        self.max_length = max_length
        self.target_length = target_length
        self.vocab_size = vocab_size
        self.seed = seed
        
        self.cache_dir = Path(cache_config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache path based on data config and tokenizer
        self.cache_path = self._get_cache_path()
        # Use model-specific tokenizer path
        self.tokenizer_path = Path(cache_config.tokenizer_path)
        if not self.tokenizer_path.is_absolute():
            self.tokenizer_path = self.cache_dir / self.tokenizer_path
        
        # Load or initialize tokenizer
        self.tokenizer = self._setup_tokenizer(str(self.tokenizer_path))
        
        # Create or load cache with tokenized sequences
        self.data_cache = self._setup_cache()
        
        # Initialize dataset length
        self._setup_indices()

    def _camel_to_lower(self, text: Optional[str]) -> str:
        """Convert CamelCase to lower case with spaces"""
        if text is None or pd.isna(text):
            return ""
        try:
            text = str(text)
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', text).lower()
        except Exception:
            return str(text)

    def _read_and_preprocess_chunks(self, config: ColumnConfig):
        """Read and preprocess data in chunks"""
        source_cols = [f'col_{i}' for i in config.source_columns]
        target_cols = [f'col_{i}' for i in config.target_columns]
        all_cols = source_cols + target_cols
        col_indices = config.source_columns + config.target_columns
        
        try:
            for chunk in pd.read_csv(
                config.file_path,
                sep=config.separator,
                usecols=col_indices,
                header=None if not config.has_header else 0,
                names=all_cols if not config.has_header else None,
                chunksize=10000,
                dtype=str,
                on_bad_lines='warn'
            ):
                if config.camel_to_lower:
                    for col_idx in config.camel_to_lower:
                        col_name = f'col_{col_idx}'
                        if col_name in chunk.columns:
                            chunk[col_name] = chunk[col_name].apply(self._camel_to_lower)
                
                source_text = chunk[source_cols].astype(str).agg(config.join_token.join, axis=1)
                target_text = chunk[target_cols].astype(str).agg(config.join_token.join, axis=1)
                yield pd.DataFrame({'source': source_text.values, 'target': target_text.values})
        except Exception as e:
            logging.error(f"Error reading file {config.file_path}: {str(e)}")
            raise

    def _get_cache_path(self) -> Path:
        """Generate cache path based on config and tokenizer"""
        config_str = str(sorted([
            (c.file_path, c.source_columns, c.target_columns, str(c.camel_to_lower)) 
            for c in self.configs
        ])) + str(self.tokenizer_path) + str(self.vocab_size)
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"dataset_cache_{cache_hash}.{self.cache_config.cache_format}"

    def _validate_cache(self, cache: h5py.File) -> bool:
        """Validate cache structure"""
        required_datasets = ['source_ids', 'target_ids', 'lengths']
        if not all(ds in cache for ds in required_datasets) or 'num_sequences' not in cache.attrs:
            logging.warning("Invalid cache structure, regenerating cache")
            return False
        return True

    def _setup_cache(self) -> h5py.File:
        """Load or create cache"""
        if self.cache_config.enable_cache and self.cache_path.exists():
            cache = h5py.File(self.cache_path, 'r')
            if self._validate_cache(cache):
                logging.info(f"Loaded existing cache from {self.cache_path}")
                return cache
            cache.close()
            self.cache_path.unlink()
        
        logging.info(f"Creating new cache at {self.cache_path}")
        return self._create_cache()

    def _create_cache(self) -> h5py.File:
        """Create new cache with tokenized data"""
        all_source_ids, all_target_ids, all_lengths = [], [], []
        
        for config in self.configs:
            for chunk in self._read_and_preprocess_chunks(config):
                if chunk.empty:
                    continue
                source_texts = chunk['source'].tolist()
                target_texts = ["[BOS] " + text + " [EOS]" for text in chunk['target'].tolist()]
                
                source_encodings = self.tokenizer.encode_batch(source_texts)
                target_encodings = self.tokenizer.encode_batch(target_texts)
                for src, tgt in zip(source_encodings, target_encodings):
                    if src.ids and tgt.ids:  # Ensure non-empty sequences
                        all_source_ids.append(src.ids)
                        all_target_ids.append(tgt.ids)
                        all_lengths.append([len(src.ids), len(tgt.ids)])
        
        if not all_source_ids:
            raise ValueError("No valid data found after preprocessing")
        
        with h5py.File(self.cache_path, 'w') as f:
            dt = h5py.special_dtype(vlen=np.dtype('int32'))
            f.create_dataset('source_ids', (len(all_source_ids),), dtype=dt)
            f.create_dataset('target_ids', (len(all_target_ids),), dtype=dt)
            f.create_dataset('lengths', (len(all_lengths), 2), dtype=np.int32)
            
            for i, (src_ids, tgt_ids, lengths) in enumerate(zip(all_source_ids, all_target_ids, all_lengths)):
                f['source_ids'][i] = src_ids
                f['target_ids'][i] = tgt_ids
                f['lengths'][i] = lengths
            
            f.attrs['num_sequences'] = len(all_source_ids)
        
        return h5py.File(self.cache_path, 'r')

    def _setup_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        """Load or train tokenizer"""
        if Path(tokenizer_path).exists():
            tokenizer = Tokenizer.from_file(tokenizer_path)
            for token in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
                if tokenizer.token_to_id(token) is None:
                    raise ValueError(f"Tokenizer at {tokenizer_path} missing special token: {token}")
            logging.info(f"Loaded tokenizer from {tokenizer_path}")
            return tokenizer
        
        logging.info(f"Training new tokenizer, saving to {tokenizer_path}")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2
        )
        
        def text_iterator():
            for config in self.configs:
                for chunk in self._read_and_preprocess_chunks(config):
                    if not chunk.empty:
                        yield from chunk['source'].tolist()
                        yield from chunk['target'].tolist()
        
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
        return tokenizer

    def get_vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer"""
        return self.tokenizer.get_vocab_size()

    def get_special_token_ids(self) -> Dict[str, int]:
        """Return special token IDs"""
        return {
            "pad_id": self.tokenizer.token_to_id("[PAD]"),
            "unk_id": self.tokenizer.token_to_id("[UNK]"),
            "bos_id": self.tokenizer.token_to_id("[BOS]"),
            "eos_id": self.tokenizer.token_to_id("[EOS]")
        }

    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve a single item from the dataset"""
        source_ids = self.data_cache['source_ids'][idx]
        target_ids = self.data_cache['target_ids'][idx]
        src_len, tgt_len = self.data_cache['lengths'][idx]

        source_tensor = torch.tensor(source_ids[:src_len], dtype=torch.long)
        target_tensor = torch.tensor(target_ids[:tgt_len], dtype=torch.long)
        
        source_tensor = self._adjust_sequence(source_tensor, self.max_length)
        target_tensor = self._adjust_sequence(target_tensor, self.target_length)
        
        return {'source_text': source_tensor, 'target_text': target_tensor}

    def _adjust_sequence(self, tensor: torch.Tensor, desired_length: int) -> torch.Tensor:
        """Pad or truncate sequence to desired length"""
        if tensor.size(0) > desired_length:
            return tensor[:desired_length]
        pad_id = self.tokenizer.token_to_id("[PAD]")
        padding = torch.full((desired_length - tensor.size(0),), pad_id, dtype=torch.long)
        return torch.cat([tensor, padding])

    def _setup_indices(self):
        """Initialize dataset length and indices"""
        self.length = self.data_cache.attrs['num_sequences']
        if self.length == 0:
            raise ValueError("Dataset is empty after caching")

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    source_texts = [item['source_text'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    
    source_padded = torch.stack(source_texts)
    target_padded = torch.stack(target_texts)
    
    return {
        'source_text': source_padded,
        'target_text': target_padded
    }
