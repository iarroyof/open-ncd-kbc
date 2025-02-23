# src/trainers/attention_lstm_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import List, Optional, Dict
import wandb
from ..data.tsv_text2text_dataset import CachedTSVDataset, ColumnConfig, CacheConfig
from ..models.text2text_autoencoders import AttentionLSTMSeq2Seq
from ..metrics.evaluation import TextGenerationMetrics

class AttentionLSTMTrainer:
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
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize datasets
        self.train_dataset = CachedTSVDataset(
            configs=train_configs,
            cache_config=CacheConfig(enable_cache=True, cache_dir=cache_dir, cache_format='h5'),
            tokenizer_path=tokenizer_path,
            vocab_size=model_config.get('vocab_size', 32000),
            max_length=model_config.get('max_seq_len', 512)
        )
        self.valid_dataset = CachedTSVDataset(
            configs=valid_configs,
            cache_config=CacheConfig(enable_cache=True, cache_dir=cache_dir, cache_format='h5'),
            tokenizer_path=tokenizer_path,
            vocab_size=model_config.get('vocab_size', 32000),
            max_length=model_config.get('max_seq_len', 512)
        )
        
        # Update vocab size
        self.model_config['vocab_size'] = self.train_dataset.get_vocab_size()
        
        # Initialize model
        self.model = AttentionLSTMSeq2Seq(**self.model_config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.0)
        )
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Metrics
        self.metrics = TextGenerationMetrics(self.train_dataset.tokenizer)
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="lstm-seq2seq", config={"model_config": model_config, "training_config": training_config})

    def _collate_fn(self, batch):
        source_ids = torch.stack([item['source_text'] for item in batch])
        target_ids = torch.stack([item['target_text'] for item in batch])
        return {'source_text': source_ids, 'target_text': target_ids}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        valid_batches = 0
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / self.training_config['num_epochs']))
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            try:
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(src=source_ids, tgt=target_ids, teacher_forcing_ratio=teacher_forcing_ratio)
                
                loss = self.criterion(outputs.view(-1, self.model_config['vocab_size']), target_ids.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                if self.use_wandb:
                    wandb.log({'batch_loss': loss.item()})
            except Exception as e:
                logging.warning(f"Error in training batch: {str(e)}")
                continue
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        valid_batches = 0
        
        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            try:
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                loss = self.criterion(outputs.view(-1, self.model_config['vocab_size']), target_ids.view(-1))
                
                total_loss += loss.item()
                valid_batches += 1
                
                predictions = outputs.argmax(dim=-1)
                mask = target_ids != 0
                for pred, ref, m in zip(predictions, target_ids, mask):
                    valid_len = m.sum().item()
                    if valid_len > 0:
                        all_predictions.append(pred[:valid_len].cpu())
                        all_references.append(ref[:valid_len].cpu())
            except Exception as e:
                logging.warning(f"Error in evaluation batch: {str(e)}")
                continue
        
        if not all_predictions:
            logging.error("No valid predictions during evaluation")
            return {'val_loss': float('inf'), 'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0}
        
        metrics = self.metrics.compute_metrics(all_predictions, all_references)
        metrics['val_loss'] = total_loss / valid_batches if valid_batches > 0 else float('inf')
        return metrics

    def train(self):
        logging.info("Starting training")
        for epoch in range(self.training_config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, "
                         f"BLEU: {val_metrics['bleu']:.4f}, ROUGE-L: {val_metrics['rougeL']:.4f}, METEOR: {val_metrics['meteor']:.4f}")
            if self.use_wandb:
                wandb.log({'epoch': epoch, 'train_loss': train_loss, **val_metrics})
        logging.info("Training completed")
        if self.use_wandb:
            wandb.finish()

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
