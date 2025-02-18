# src/trainers/positional_autoencoder_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import List, Optional, Dict
import wandb
from tqdm import tqdm
import gc
import os

from ..data.tsv_text2text_dataset import (
    CachedTSVDataset, 
    ColumnConfig, 
    CacheConfig,
    collate_fn
)
from ..models.text2text_autoencoders import PositionalAutoencoder
from ..metrics.evaluation import TextGenerationMetrics

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AutoencoderTrainer:
    def __init__(
        self,
        model_config: dict,
        training_config: dict,
        train_configs: List[ColumnConfig],
        valid_configs: List[ColumnConfig],
        tokenizer_path: Optional[str] = None,
        cache_dir: str = "./cache",
        log_dir: str = "./logs",
        use_wandb: bool = False
    ):
        self.model_config = model_config.copy()
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "training.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Cache configuration
        cache_config = CacheConfig(
            enable_cache=True,
            cache_dir=cache_dir,
            cache_format='h5'
        )
        
        # Initialize datasets
        logging.info("Initializing train dataset")
        self.train_dataset = CachedTSVDataset(
            configs=train_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('max_seq_len', 512)
        )
        
        logging.info("Initializing validation dataset")
        self.valid_dataset = CachedTSVDataset(
            configs=valid_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('max_seq_len', 512)
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True,
            prefetch_factor=2
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Update model_config with vocab_size from dataset
        self.model_config['vocab_size'] = self.train_dataset.get_vocab_size()
        
        # Initialize model
        self.model = PositionalAutoencoder(**self.model_config).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        # Cosine annealing scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=training_config['learning_rate'],
            epochs=training_config['num_epochs'],
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1
        )
        
        # Initialize metrics calculator
        self.metrics = TextGenerationMetrics(self.train_dataset.tokenizer)
        
        # Initialize wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="positional-autoencoder",
                config={
                    "model_config": self.model_config,
                    "training_config": training_config
                }
            )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Ensure target has correct sequence length
                if target_ids.size(1) != self.model_config['target_seq_len']:
                    target_ids = target_ids[:, :self.model_config['target_seq_
