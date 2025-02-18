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

# src/trainers/positional_autoencoder_trainer.py
from ..data.tsv_text2text_dataset import (
    CachedTSVDataset, 
    ColumnConfig, 
    CacheConfig,
    collate_fn
)
from ..models.text2text_autoencoders import PositionalAutoencoder
from ..metrics import TextGenerationMetrics

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
        
        # Initialize datasets with caching
        logging.info("Initializing train dataset (will cache if not cached)")
        self.train_dataset = CachedTSVDataset(
            configs=train_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000)
        )
        
        logging.info("Initializing validation dataset (will cache if not cached)")
        self.valid_dataset = CachedTSVDataset(
            configs=valid_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000)
        )
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True,
            prefetch_factor=2
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
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
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Forward pass
                outputs = self.model(source_ids)
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                if self.use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
                    
                # Clear cache periodically
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Forward pass
                outputs = self.model(source_ids)
                
                # Calculate loss
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                total_loss += loss.item()
                
                # Get predictions
                predictions = outputs.argmax(dim=-1)
                all_predictions.append(predictions.cpu())
                all_references.append(target_ids.cpu())
                
            except Exception as e:
                print(f"Error in evaluation batch: {str(e)}")
                continue
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_references = torch.cat(all_references, dim=0)
        
        metrics = self.metrics.compute_metrics(all_predictions, all_references)
        metrics['val_loss'] = total_loss / len(self.valid_loader)
        
        # Clear memory
        del all_predictions, all_references
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics

    def train(self):
        logging.info("Starting training")
        best_metrics = {'val_loss': float('inf'), 'bleu': 0}
        
        for epoch in range(self.training_config['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Log metrics
            log_msg = f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}"
            log_msg += f", BLEU: {val_metrics['bleu']:.4f}"
            log_msg += f", ROUGE-L: {val_metrics['rougeL']:.4f}"
            log_msg += f", METEOR: {val_metrics['meteor']:.4f}"
            logging.info(log_msg)
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics
                })
            
            # Save best model
            if val_metrics['val_loss'] < best_metrics['val_loss'] or \
               val_metrics['bleu'] > best_metrics['bleu']:
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics)
                logging.info(f"Saved new best model with metrics: {val_metrics}")

        logging.info("Training completed")
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        torch.save(checkpoint, self.log_dir / f'checkpoint_epoch_{epoch}.pt')
