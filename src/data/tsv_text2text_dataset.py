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

class CachedTSVDataset(Dataset):
    def __init__(
        self,
        configs: List[ColumnConfig],
        cache_config: CacheConfig,
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        max_length: int = 512,  # Model-specific max_seq_len
        target_length: int = 64,  # Model-specific target_seq_len
        seed: int = 42
    ):
        self.configs = configs
        self.cache_config = cache_config
        self.max_length = max_length
        self.target_length = target_length
        self.vocab_size = vocab_size
        
        self.cache_dir = Path(cache_config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Single cache path based on data config (not model-specific lengths)
        self.cache_path = self._get_cache_path()
        self.tokenizer_path = self.cache_dir / "tokenizer.json"  # Single tokenizer for all models
        
        # Load or initialize tokenizer
        self.tokenizer = self._setup_tokenizer(tokenizer_path if tokenizer_path else str(self.tokenizer_path))
        
        # Create or load cache with raw tokenized sequences
        self.data_cache = self._setup_cache()
        
        # Initialize indices
        self._setup_indices()

    def _camel_to_lower(self, text: Optional[str]) -> str:
        """Convert CamelCase to lower case with spaces, handling non-string inputs"""
        if text is None or pd.isna(text):
            return ""  # Return empty string for None/NaN
        try:
            text = str(text)  # Convert to string if possible (e.g., int/float)
            return re.sub(r'(?<!^)(?=[A-Z])', ' ', text).lower()
        except Exception as e:
            logging.warning(f"Failed to convert text to lower case: {str(e)}")
            return str(text)  #Fallback to original as string

    def _read_and_preprocess_chunks(self, config: ColumnConfig):
        """Read data in chunks and apply preprocessing including CamelCase conversion"""
        try:
            source_cols = [f'col_{i}' for i in config.source_columns]
            target_cols = [f'col_{i}' for i in config.target_columns]
            all_cols = source_cols + target_cols
            col_indices = config.source_columns + config.target_columns
            
            for chunk in pd.read_csv(
                config.file_path,
                sep=config.separator,
                usecols=col_indices,
                header=None,
                names=all_cols,
                chunksize=10000,
                dtype=str,
                on_bad_lines='warn'  # Log bad lines instead of skipping
            ):
                # Log chunk for debugging
                logging.debug(f"Processing chunk with columns: {chunk.columns}")
                
                # Apply CamelCase transformation
                if config.camel_to_lower:
                    for col_idx in config.camel_to_lower:
                        col_name = f'col_{col_idx}'
                        if col_name in chunk.columns:
                            chunk[col_name] = chunk[col_name].apply(self._camel_to_lower)
                        else:
                            logging.warning(f"Column {col_name} not found in chunk")
                
                source_text = chunk[source_cols].astype(str).agg(config.join_token.join, axis=1)
                target_text = chunk[target_cols].astype(str).agg(config.join_token.join, axis=1)
                processed_chunk = pd.DataFrame({'source': source_text.values, 'target': target_text.values})
                yield processed_chunk
        except Exception as e:
            logging.error(f"Error reading file {config.file_path}: {str(e)}")
            yield pd.DataFrame(columns=['source', 'target'])

    def _get_cache_path(self) -> Path:
        """Generate cache path based on dataset configuration (excluding model-specific lengths)"""
        config_str = str(sorted([
            (c.file_path, c.source_columns, c.target_columns, str(c.camel_to_lower)) 
            for c in self.configs
        ]))
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()
        return self.cache_dir / f"dataset_cache_{cache_hash}.{self.cache_config.cache_format}"

    def _validate_cache(self, cache: h5py.File) -> bool:
        """Validate cache file structure and contents"""
        try:
            if 'source_ids' not in cache or 'target_ids' not in cache or 'lengths' not in cache:
                logging.error("Cache missing required datasets")
                return False
            
            required_attrs = ['num_sequences']
            for attr in required_attrs:
                if attr not in cache.attrs:
                    logging.error(f"Cache missing required attribute: {attr}")
                    return False
            
            if len(cache['source_ids'].shape) != 1 or len(cache['target_ids'].shape) != 1 or len(cache['lengths'].shape) != 2:
                logging.error("Invalid data shapes in cache")
                return False
            
            if cache['source_ids'].shape[0] != cache['target_ids'].shape[0] or cache['source_ids'].shape[0] != cache['lengths'].shape[0]:
                logging.error("Mismatched sequence counts in cache")
                return False
            
            try:
                cache['source_ids'][0]
                cache['target_ids'][0]
                cache['lengths'][0]
            except Exception as e:
                logging.error(f"Failed to read data from cache: {str(e)}")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating cache: {str(e)}")
            return False

    def _setup_cache(self) -> h5py.File:
        """Setup cache file with raw tokenized sequences"""
        try:
            if self.cache_path.exists() and self.tokenizer_path.exists():
                logging.info(f"Loading existing cache from {self.cache_path}")
                cache = h5py.File(self.cache_path, 'r')
                if self._validate_cache(cache):
                    logging.info("Cache validated successfully")
                    return cache
                else:
                    cache.close()
                    logging.warning("Cache validation failed, will recreate cache")
                    self.cache_path.unlink()
                    self.tokenizer_path.unlink()
            
            return self._create_cache()
        except Exception as e:
            logging.error(f"Error setting up cache: {str(e)}")
            if self.cache_path.exists():
                self.cache_path.unlink()
            if self.tokenizer_path.exists():
                self.tokenizer_path.unlink()
            try:
                logging.info("Attempting to create new cache")
                return self._create_cache()
            except Exception as create_error:
                raise RuntimeError(f"Failed to create new cache: {str(create_error)}") from create_error

    def _create_cache(self) -> h5py.File:
        logging.info(f"Creating cache file at {self.cache_path}")
        
        try:
            all_source_ids = []
            all_target_ids = []
            all_lengths = []  # Store [source_len, target_len] for each sequence
            
            logging.info("Processing and tokenizing sequences")
            for config in self.configs:
                for chunk in self._read_and_preprocess_chunks(config):
                    if chunk.empty:
                        continue
                    source_texts = chunk['source'].tolist()
                    target_texts = chunk['target'].tolist()
                    
                    # Add [BOS] and [EOS] to target texts
                    target_texts_with_tokens = ["[BOS] " + text + " [EOS]" for text in target_texts]
                    
                    source_encodings = self.tokenizer.encode_batch(source_texts)
                    target_encodings = self.tokenizer.encode_batch(target_texts_with_tokens)
                    for src, tgt in zip(source_encodings, target_encodings):
                        if src and tgt:
                            src_ids = src.ids
                            tgt_ids = tgt.ids
                            all_source_ids.append(src_ids)
                            all_target_ids.append(tgt_ids)
                            all_lengths.append([len(src_ids), len(tgt_ids)])
            
            if not all_source_ids or not all_target_ids:
                raise ValueError("No valid sequences found in the dataset")
            
            # Convert to HDF5-compatible ragged arrays
            logging.info("Converting to HDF5 arrays")
            with h5py.File(self.cache_path, 'w') as f:
                dt = h5py.special_dtype(vlen=np.dtype('int32'))
                source_ds = f.create_dataset('source_ids', (len(all_source_ids),), dtype=dt)
                target_ds = f.create_dataset('target_ids', (len(all_target_ids),), dtype=dt)
                lengths_ds = f.create_dataset('lengths', (len(all_lengths), 2), dtype=np.int32)
                
                for i, (src_ids, tgt_ids, lengths) in enumerate(zip(all_source_ids, all_target_ids, all_lengths)):
                    source_ds[i] = src_ids
                    target_ds[i] = tgt_ids
                    lengths_ds[i] = lengths
                
                f.attrs['num_sequences'] = len(all_source_ids)
            
            self.tokenizer.save(str(self.tokenizer_path))
            logging.info(f"Saved tokenizer to {self.tokenizer_path}")
            return h5py.File(self.cache_path, 'r')
        except Exception as e:
            logging.error(f"Error creating cache: {str(e)}")
            if self.cache_path.exists():
                self.cache_path.unlink()
            raise RuntimeError(f"Failed to create cache: {str(e)}") from e

    def _setup_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        if Path(tokenizer_path).exists():
            logging.info(f"Loading tokenizer from {tokenizer_path}")
            return Tokenizer.from_file(tokenizer_path)
        
        logging.info("Creating and training new tokenizer")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
            min_frequency=2
        )
        
        def text_iterator():
            all_texts = []
            for config in self.configs:
                for chunk in self._read_and_preprocess_chunks(config):
                    if not chunk.empty:
                        texts = chunk['source'].tolist() + chunk['target'].tolist()
                        texts = [str(text) for text in texts if text is not None and str(text).strip()]
                        all_texts.extend(texts)
            return all_texts
        
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        return tokenizer

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.length

    def __del__(self):
        if hasattr(self, 'data_cache') and isinstance(self.data_cache, h5py.File):
            try:
                self.data_cache.close()
            except Exception as e:
                logging.error(f"Error closing cache file: {str(e)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            source_ids = self.data_cache['source_ids'][idx]
            target_ids = self.data_cache['target_ids'][idx]
            src_len, tgt_len = self.data_cache['lengths'][idx]

            # Pad/truncate to model-specific lengths at runtime
            source_tensor = torch.tensor(source_ids[:src_len], dtype=torch.long)
            target_tensor = torch.tensor(target_ids[:tgt_len], dtype=torch.long)
            
            source_tensor = self._adjust_sequence(source_tensor, self.max_length, pad_left=True)
            target_tensor = self._adjust_sequence(target_tensor, self.target_length, pad_left=False)
            
            return {'source_text': source_tensor, 'target_text': target_tensor}
        except Exception as e:
            logging.error(f"Error reading item {idx}: {str(e)}")
            return {
                'source_text': torch.zeros(self.max_length, dtype=torch.long),
                'target_text': torch.zeros(self.target_length, dtype=torch.long)
            }

    def _adjust_sequence(self, tensor: torch.Tensor, desired_length: int, pad_left: bool = False) -> torch.Tensor:
        current_length = tensor.size(0)
        if current_length == desired_length:
            return tensor
        if current_length > desired_length:
            return tensor[-desired_length:] if pad_left else tensor[:desired_length]
        padding_size = desired_length - current_length
        padding = torch.zeros(padding_size, dtype=tensor.dtype)
        return torch.cat([padding, tensor] if pad_left else [tensor, padding])

    def _setup_indices(self):
        try:
            self.length = self.data_cache.attrs.get('num_sequences', len(self.data_cache['source_ids']))
            if self.length == 0:
                raise ValueError("Dataset contains no sequences")
            logging.info(f"Dataset contains {self.length} sequences")
        except Exception as e:
            logging.error(f"Error setting up indices: {str(e)}")
            raise RuntimeError(f"Failed to initialize dataset: {str(e)}") from e

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }
