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

    def _validate_cache(self, cache: h5py.File) -> bool:
        """Validate cache file structure and contents"""
        try:
            # Check required datasets exist
            if 'source_ids' not in cache or 'target_ids' not in cache:
                logging.error("Cache missing required datasets")
                return False
                
            # Check required attributes
            required_attrs = ['max_source_len', 'max_target_len', 'num_sequences']
            for attr in required_attrs:
                if attr not in cache.attrs:
                    logging.error(f"Cache missing required attribute: {attr}")
                    return False
                    
            # Check data shapes
            if len(cache['source_ids'].shape) != 2 or len(cache['target_ids'].shape) != 2:
                logging.error("Invalid data shapes in cache")
                return False
                
            # Check number of sequences matches
            if cache['source_ids'].shape[0] != cache['target_ids'].shape[0]:
                logging.error("Mismatched sequence counts in cache")
                return False
                
            # Check data is readable
            try:
                cache['source_ids'][0]
                cache['target_ids'][0]
            except Exception as e:
                logging.error(f"Failed to read data from cache: {str(e)}")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating cache: {str(e)}")
            return False

    def _setup_cache(self) -> Union[h5py.File, np.memmap]:
        """Setup cache file based on configuration"""
        try:
            if self.cache_path.exists():
                logging.info(f"Loading existing cache from {self.cache_path}")
                if self.cache_config.cache_format == 'h5':
                    cache = h5py.File(self.cache_path, 'r')
                    if self._validate_cache(cache):
                        return cache
                    else:
                        cache.close()
                        logging.warning("Cache validation failed, will recreate cache")
                        self.cache_path.unlink()
                else:  # mmap
                    return np.load(self.cache_path, mmap_mode='r')
            
            # Create new cache
            return self._create_cache()
                
        except Exception as e:
            logging.error(f"Error setting up cache: {str(e)}")
            # If cache exists but is corrupted, remove it
            if self.cache_path.exists():
                logging.info(f"Removing corrupted cache file: {self.cache_path}")
                self.cache_path.unlink()
            # Try to create new cache
            try:
                logging.info("Attempting to create new cache")
                return self._create_cache()
            except Exception as create_error:
                raise RuntimeError(f"Failed to create new cache: {str(create_error)}") from create_error

    def _create_cache(self) -> Union[h5py.File, np.memmap]:
        """Create and populate cache file"""
        logging.info(f"Creating cache file at {self.cache_path}")
        
        try:
            # Initialize lists for storage
            all_source_ids = []
            all_target_ids = []
            max_source_len = 0
            max_target_len = 0
            
            # Process all data and track lengths
            logging.info("First pass: calculating maximum sequence lengths")
            for config in self.configs:
                for chunk in self._read_chunks(config):
                    if chunk.empty:
                        continue
                        
                    source_encodings = self.tokenizer.encode_batch(chunk['source'].tolist())
                    target_encodings = self.tokenizer.encode_batch(chunk['target'].tolist())
                    
                    for src, tgt in zip(source_encodings, target_encodings):
                        if src and tgt:  # Ensure both sequences exist
                            max_source_len = max(max_source_len, len(src.ids))
                            max_target_len = max(max_target_len, len(tgt.ids))

            # Clip maximum lengths to model's max_length
            max_source_len = min(max_source_len, self.max_length)
            max_target_len = min(max_target_len, self.max_length)
            
            logging.info(f"Maximum source length: {max_source_len}")
            logging.info(f"Maximum target length: {max_target_len}")
            
            # Second pass: create padded arrays
            logging.info("Second pass: creating padded sequences")
            for config in self.configs:
                for chunk in self._read_chunks(config):
                    if chunk.empty:
                        continue
                        
                    source_encodings = self.tokenizer.encode_batch(chunk['source'].tolist())
                    target_encodings = self.tokenizer.encode_batch(chunk['target'].tolist())
                    
                    for src, tgt in zip(source_encodings, target_encodings):
                        if src and tgt:  # Ensure both sequences exist
                            # Pad or truncate source sequence
                            src_ids = src.ids[:max_source_len]
                            src_ids = src_ids + [0] * (max_source_len - len(src_ids))
                            
                            # Pad or truncate target sequence
                            tgt_ids = tgt.ids[:max_target_len]
                            tgt_ids = tgt_ids + [0] * (max_target_len - len(tgt_ids))
                            
                            all_source_ids.append(src_ids)
                            all_target_ids.append(tgt_ids)
            
            if not all_source_ids or not all_target_ids:
                raise ValueError("No valid sequences found in the dataset")
            
            # Convert to numpy arrays
            logging.info("Converting to numpy arrays")
            source_array = np.array(all_source_ids, dtype=np.int32)
            target_array = np.array(all_target_ids, dtype=np.int32)
            
            logging.info(f"Final arrays shape - Source: {source_array.shape}, Target: {target_array.shape}")
            
            # Create cache file
            if self.cache_config.cache_format == 'h5':
                logging.info("Creating HDF5 cache file")
                with h5py.File(self.cache_path, 'w') as f:
                    # Store the sequences
                    f.create_dataset('source_ids', data=source_array)
                    f.create_dataset('target_ids', data=target_array)
                    
                    # Store metadata
                    f.attrs['max_source_len'] = max_source_len
                    f.attrs['max_target_len'] = max_target_len
                    f.attrs['num_sequences'] = len(all_source_ids)
                
                logging.info("Cache file created successfully")
                return h5py.File(self.cache_path, 'r')  # Reopen in read mode
                
            else:
                logging.info("Creating numpy memory-mapped cache file")
                data = np.array(list(zip(all_source_ids, all_target_ids)), dtype=np.int32)
                np.save(self.cache_path, data)
                return np.load(self.cache_path, mmap_mode='r')
                
        except Exception as e:
            logging.error(f"Error creating cache: {str(e)}")
            # Clean up partial cache file if it exists
            if self.cache_path.exists():
                self.cache_path.unlink()
            raise RuntimeError(f"Failed to create cache: {str(e)}") from e

    def _setup_indices(self):
        """Setup indices for dataset access"""
        try:
            if self.cache_config.cache_format == 'h5':
                if 'num_sequences' in self.data_cache.attrs:
                    self.length = self.data_cache.attrs['num_sequences']
                else:
                    self.length = len(self.data_cache['source_ids'])
            else:
                self.length = len(self.data_cache)
                
            if self.length == 0:
                raise ValueError("Dataset contains no sequences")
                
            logging.info(f"Dataset contains {self.length} sequences")
            
        except Exception as e:
            logging.error(f"Error setting up indices: {str(e)}")
            raise RuntimeError(f"Failed to initialize dataset: {str(e)}") from e

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

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'data_cache'):
            if isinstance(self.data_cache, h5py.File):
                try:
                    self.data_cache.close()
                except Exception as e:
                    logging.error(f"Error closing cache file: {str(e)}")
                    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        try:
            if self.cache_config.cache_format == 'h5':
                source_ids = self.data_cache['source_ids'][idx].copy()
                target_ids = self.data_cache['target_ids'][idx].copy()
            else:
                source_ids, target_ids = self.data_cache[idx]

            # Convert to tensors
            source_tensor = torch.tensor(source_ids, dtype=torch.long)
            target_tensor = torch.tensor(target_ids, dtype=torch.long)
            
            # Verify shapes
            if source_tensor.size(0) != self.data_cache.attrs['max_source_len']:
                source_tensor = self._adjust_sequence(
                    source_tensor, 
                    self.data_cache.attrs['max_source_len'],
                    pad_left=True  # Pad from left for source
                )
            
            if target_tensor.size(0) != self.data_cache.attrs['max_target_len']:
                target_tensor = self._adjust_sequence(
                    target_tensor, 
                    self.data_cache.attrs['max_target_len'],
                    pad_left=False  # Pad from right for target
                )

            return {
                'source_text': source_tensor,
                'target_text': target_tensor
            }
            
        except Exception as e:
            logging.error(f"Error reading item {idx}: {str(e)}")
            # Return zero tensors of correct shape
            return {
                'source_text': torch.zeros(self.data_cache.attrs['max_source_len'], dtype=torch.long),
                'target_text': torch.zeros(self.data_cache.attrs['max_target_len'], dtype=torch.long)
            }
            
    def _adjust_sequence(self, tensor: torch.Tensor, desired_length: int, pad_left: bool = False) -> torch.Tensor:
        """Adjust sequence length by padding or truncating"""
        current_length = tensor.size(0)
        
        if current_length == desired_length:
            return tensor
            
        if current_length > desired_length:
            # Truncate
            if pad_left:
                return tensor[-desired_length:]  # Keep right side
            else:
                return tensor[:desired_length]  # Keep left side
        else:
            # Pad
            padding_size = desired_length - current_length
            if pad_left:
                return torch.cat([torch.zeros(padding_size, dtype=tensor.dtype), tensor])
            else:
                return torch.cat([tensor, torch.zeros(padding_size, dtype=tensor.dtype)])

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for padding sequences"""
    # All sequences in the batch should already be padded to the same length
    # Just stack them
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
