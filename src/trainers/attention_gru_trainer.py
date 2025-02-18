# src/trainers/attention_gru_trainer.py

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
from ..models.attention_gru import AttentionGRUModel
from ..metrics.evaluation import TextGenerationMetrics

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AttentionGRUTrainer:
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
        self.model = AttentionGRUModel(**self.model_config).to(self.device)
        
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
                project="attention-gru",
                config={
                    "model_config": self.model_config,
                    "training_config": training_config
                }
            )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with teacher forcing"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # Calculate teacher forcing ratio (can decay over epochs)
        teacher_forcing_ratio = max(
            0.0,
            1.0 - (epoch / self.training_config['num_epochs'])
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Ensure target has correct sequence length
                if target_ids.size(1) != self.model_config['target_seq_len']:
                    if target_ids.size(1) > self.model_config['target_seq_len']:
                        target_ids = target_ids[:, :self.model_config['target_seq_len']]
                    else:
                        pad_len = self.model_config['target_seq_len'] - target_ids.size(1)
                        target_ids = torch.nn.functional.pad(target_ids, (0, pad_len), value=0)
                
                # Forward pass with teacher forcing
                outputs = self.model(
                    src=source_ids,
                    tgt=target_ids,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                
                # Calculate loss
                outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                targets_flat = target_ids.contiguous().view(-1)
                loss = nn.CrossEntropyLoss(ignore_index=0)(outputs_flat, targets_flat)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                valid_batches += 1
                avg_loss = total_loss / valid_batches
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'tf_ratio': f'{teacher_forcing_ratio:.2f}'
                })
                
                if self.use_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'teacher_forcing_ratio': teacher_forcing_ratio
                    })
                
                # Periodic memory cleanup
                if batch_idx % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.warning(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model without teacher forcing"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        valid_batches = 0
        
        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Ensure target has correct sequence length
                if target_ids.size(1) != self.model_config['target_seq_len']:
                    if target_ids.size(1) > self.model_config['target_seq_len']:
                        target_ids = target_ids[:, :self.model_config['target_seq_len']]
                    else:
                        pad_len = self.model_config['target_seq_len'] - target_ids.size(1)
                        target_ids = torch.nn.functional.pad(target_ids, (0, pad_len), value=0)
                
                # Forward pass without teacher forcing
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                
                # Calculate loss
                outputs_flat = outputs.contiguous().view(-1, outputs.size(-1))
                targets_flat = target_ids.contiguous().view(-1)
                loss = nn.CrossEntropyLoss(ignore_index=0)(outputs_flat, targets_flat)
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Get predictions
                predictions = outputs.argmax(dim=-1)  # [batch_size, target_seq_len]
                
                # Store predictions and references
                all_predictions.append(predictions.cpu())
                all_references.append(target_ids.cpu())
                
            except Exception as e:
                logging.warning(f"Error in evaluation batch: {str(e)}")
                continue
        
        if not all_predictions:
            logging.error("No valid predictions during evaluation")
            return {
                'val_loss': float('inf'),
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'meteor': 0.0
            }
        
        try:
            # Compute average loss
            avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
            
            # Concatenate all predictions and references
            all_predictions = torch.cat(all_predictions, dim=0)
            all_references = torch.cat(all_references, dim=0)
            
            # Compute metrics
            metrics = self.metrics.compute_metrics(all_predictions, all_references)
            metrics['val_loss'] = avg_loss
            
        except Exception as e:
            logging.error(f"Error computing metrics: {str(e)}")
            metrics = {
                'val_loss': avg_loss if 'avg_loss' in locals() else float('inf'),
                'bleu': 0.0,
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'meteor': 0.0
            }
        
        # Clear memory
        del all_predictions, all_references
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
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

    def train(self):
        """Complete training loop"""
        logging.info("Starting training")
        best_metrics = {'val_loss': float('inf'), 'bleu': 0}
        
        try:
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
                    
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            logging.info("Training completed")
            if self.use_wandb:
                wandb.finish()

    def __del__(self):
        """Cleanup resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
