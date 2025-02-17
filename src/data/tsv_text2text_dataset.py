import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import itertools
import re
import csv
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

def convert_camel_case(text: str) -> str:
    """Convert CamelCase to space-separated lowercase text"""
    if not isinstance(text, str):
        text = str(text)
    # Handle special cases first (e.g., XMLHTTPRequest)
    text = re.sub(r'([A-Z]{2,})', lambda m: ' ' + m.group(1).lower() + ' ', text)
    # Handle regular CamelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Clean up any double spaces and convert to lowercase
    return ' '.join(text.lower().split())

@dataclass
class ColumnConfig:
    """Configuration for column processing"""
    file_path: str
    source_columns: List[Union[str, int]]
    target_columns: List[Union[str, int]]
    has_header: bool = True
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
        self.configs = configs
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.vocab_size = vocab_size  # Store vocab_size for later use
        
        np.random.seed(seed)
        
        # Initialize or load tokenizer
        self.tokenizer = self._setup_tokenizer(tokenizer_path, vocab_size)
        
        # Calculate total size and create index mapping
        self.total_size = 0
        self.file_indices = []
        
        for config in configs:
            file_size = self._count_valid_rows(config)
            self.file_indices.extend([
                (self.total_size + i, config) 
                for i in range(file_size)
            ])
            self.total_size += file_size
        
        np.random.shuffle(self.file_indices)

    def _read_specific_columns(self, file_path: str, columns: List[int], 
                             chunk_size: int, separator: str, has_header: bool):
        """Read only specified columns from file"""
        try:
            for chunk in pd.read_csv(
                file_path,
                sep=separator,
                header=0 if has_header else None,
                usecols=columns,
                names=columns,
                engine='python',
                on_bad_lines='skip',
                dtype=str,
                chunksize=chunk_size,
                quoting=csv.QUOTE_NONE,  # Disable quote handling
                encoding='utf-8'
            ):
                yield chunk
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            yield pd.DataFrame()

    def _count_valid_rows(self, config: ColumnConfig) -> int:
        """Count valid rows containing required columns"""
        required_columns = list(set(config.source_columns + config.target_columns))
        count = 0
        
        for chunk in self._read_specific_columns(
            config.file_path,
            required_columns,
            self.chunk_size,
            config.separator,
            config.has_header
        ):
            if not chunk.empty:
                valid_rows = chunk.notna().all(axis=1)
                count += valid_rows.sum()
        
        return count - (1 if config.has_header else 0)

    def _process_text(self, text: str) -> str:
        """Process text by converting CamelCase and cleaning"""
        return convert_camel_case(text)

    def _process_dataframe_text(self, df: pd.DataFrame, columns: List[int]) -> pd.DataFrame:
        """Process text in specified columns of a dataframe"""
        result = df.copy()
        for col in columns:
            result[col] = result[col].map(self._process_text)
        return result

    def _setup_tokenizer(self, tokenizer_path: Optional[str], vocab_size: int) -> Tokenizer:
        if tokenizer_path and Path(tokenizer_path).exists():
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return tokenizer
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        )
        
        def text_iterator():
            for config in self.configs:
                required_columns = list(set(config.source_columns + config.target_columns))
                
                for chunk in self._read_specific_columns(
                    config.file_path,
                    required_columns,
                    self.chunk_size,
                    config.separator,
                    config.has_header
                ):
                    if chunk.empty:
                        continue
                        
                    valid_rows = chunk.notna().all(axis=1)
                    chunk = chunk[valid_rows]
                    
                    all_text = []
                    for cols in [config.source_columns, config.target_columns]:
                        # Process each column's text using the new method
                        processed_chunk = self._process_dataframe_text(chunk[cols], cols)
                        text = processed_chunk.agg(config.join_token.join, axis=1)
                        all_text.extend(text.tolist())
                    
                    if all_text:
                        yield all_text
        
        tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
        
        if self.cache_dir:
            tokenizer.save(str(self.cache_dir / "tokenizer.json"))
        
        return tokenizer

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the tokenizer"""
        return self.vocab_size

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx, config = self.file_indices[idx]
        chunk_idx = global_idx // self.chunk_size
        local_idx = global_idx % self.chunk_size
        
        required_columns = list(set(config.source_columns + config.target_columns))
        
        try:
            chunk = next(itertools.islice(
                self._read_specific_columns(
                    config.file_path,
                    required_columns,
                    self.chunk_size,
                    config.separator,
                    config.has_header
                ),
                chunk_idx, chunk_idx + 1
            ))
            
            if chunk.empty:
                raise ValueError("Empty chunk")
            
            valid_rows = chunk.notna().all(axis=1)
            chunk = chunk[valid_rows].reset_index(drop=True)
            
            if local_idx >= len(chunk):
                raise IndexError("Invalid local index after filtering")
            
            # Process source and target text using the new method
            source_chunk = self._process_dataframe_text(
                chunk[config.source_columns], config.source_columns)
            target_chunk = self._process_dataframe_text(
                chunk[config.target_columns], config.target_columns)
            
            source_text = config.join_token.join(source_chunk.iloc[local_idx].tolist())
            target_text = config.join_token.join(target_chunk.iloc[local_idx].tolist())
            
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
            
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            return {
                "source_text": torch.zeros(1, dtype=torch.long),
                "target_text": torch.zeros(1, dtype=torch.long)
            }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    valid_batch = [item for item in batch if item["source_text"].size(0) > 0]
    
    if not valid_batch:
        return {
            "source_text": torch.zeros((1, 1), dtype=torch.long),
            "target_text": torch.zeros((1, 1), dtype=torch.long)
        }
    
    return {
        key: torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in valid_batch],
            batch_first=True,
            padding_value=0
        )
        for key in valid_batch[0].keys()
    }
