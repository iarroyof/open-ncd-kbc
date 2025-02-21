import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import
import math  # Added missing import
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
from ..models.text2text_autoencoders import ConvS2S
from ..metrics.evaluation import TextGenerationMetrics

class ConvS2STrainer:
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
        
        # Initialize datasets with explicit sequence lengths
        logging.info("Initializing train dataset")
        self.train_dataset = CachedTSVDataset(
            configs=train_configs,
            cache_config=CacheConfig(
                enable_cache=True,
                cache_dir=cache_dir,
                cache_format='h5'
            ),
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('max_seq_len', 512)
        )
        
        logging.info("Initializing validation dataset")
        self.valid_dataset = CachedTSVDataset(
            configs=valid_configs,
            cache_config=CacheConfig(
                enable_cache=True,
                cache_dir=cache_dir,
                cache_format='h5'
            ),
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('max_seq_len', 512)
        )
        
        # Update model config with actual vocab size
        self.model_config['vocab_size'] = self.train_dataset.get_vocab_size()
        
        # Initialize model
        self.model = ConvS2S(**self.model_config).to(self.device)
        
        # Initialize optimizer with Nesterov momentum
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            momentum=0.99,
            nesterov=True,
            weight_decay=training_config.get('weight_decay', 0.0)
        )
        
        # Initialize dataloaders with dynamic batch size
        self.batch_size = training_config['batch_size']
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._custom_collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._custom_collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=training_config.get('label_smoothing', 0.1)
        )
        
        # Initialize metrics
        self.metrics = TextGenerationMetrics(self.train_dataset.tokenizer)
        
        # Setup wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="conv-s2s",
                config={
                    "model_config": self.model_config,
                    "training_config": training_config
                }
            )

    def _custom_collate_fn(self, batch):
        """Custom collate function to ensure consistent sequence lengths"""
        # Get max lengths for this batch
        src_max_len = min(
            max(len(x['source_text']) for x in batch),
            self.model_config['max_seq_len']
        )
        tgt_max_len = min(
            max(len(x['target_text']) for x in batch),
            self.model_config['target_seq_len']
        )
        
        # Pad sequences to same length
        source_ids = torch.zeros((len(batch), src_max_len), dtype=torch.long)
        target_ids = torch.zeros((len(batch), tgt_max_len), dtype=torch.long)
        
        for i, item in enumerate(batch):
            src = torch.tensor(item['source_text'][-src_max_len:], dtype=torch.long)
            tgt = torch.tensor(item['target_text'][:tgt_max_len], dtype=torch.long)
            
            source_ids[i, :len(src)] = src
            target_ids[i, :len(tgt)] = tgt
        
        return {
            'source_text': source_ids,
            'target_text': target_ids
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        # Calculate teacher forcing ratio
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / self.training_config['num_epochs']))
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(
                    src=source_ids,
                    tgt=target_ids,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                
                # Calculate loss
                loss_mask = (target_ids != 0).float()
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                # Scale loss by sqrt(length) as per paper
                raw_loss = self.criterion(outputs_flat, targets_flat)
                valid_tokens = loss_mask.sum()
                loss = raw_loss * math.sqrt(valid_tokens)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                total_loss += batch_loss
                valid_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'tf_ratio': f'{teacher_forcing_ratio:.2f}'
                })
                
                if self.use_wandb:
                    wandb.log({
                        'batch_loss': batch_loss,
                        'teacher_forcing_ratio': teacher_forcing_ratio
                    })
                    
            except Exception as e:
                logging.warning(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        valid_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(self.valid_loader, desc="Evaluating")):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Forward pass
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                
                # Ensure output and target lengths match
                min_len = min(outputs.size(1), target_ids.size(1))
                outputs = outputs[:, :min_len, :]
                target_ids = target_ids[:, :min_len]
                
                # Calculate loss
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                total_loss += loss.item()
                valid_batches += 1
                
                # Get predictions
                predictions = outputs.argmax(dim=-1)
                
                # Remove padding
                mask = target_ids != 0
                for pred, ref, m in zip(predictions, target_ids, mask):
                    valid_len = m.sum().item()
                    if valid_len > 0:
                        all_predictions.append(pred[:valid_len])
                        all_references.append(ref[:valid_len])
                
            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
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
        
        # Calculate metrics
        metrics = self.metrics.compute_metrics(
            torch.stack(all_predictions),
            torch.stack(all_references)
        )
        metrics['val_loss'] = total_loss / valid_batches if valid_batches > 0 else float('inf')
        
        return metrics

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
