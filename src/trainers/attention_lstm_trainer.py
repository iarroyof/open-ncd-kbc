# src/trainers/attention_lstm_trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from typing import List, Optional, Dict
import wandb
import random
import numpy as np
from pathlib import Path
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
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup prediction logging
        self.predictions_log_path = self.log_dir / "model_predictions.log"
        self.prediction_logger = self._setup_prediction_logger()
        
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
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
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
        
        # Tokenizer for decoding predictions
        self.tokenizer = self.train_dataset.tokenizer
        
        # Tracking best model
        self.best_bleu = 0.0
        self.best_epoch = 0
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="lstm-seq2seq", config={"model_config": model_config, "training_config": training_config})

    def _setup_prediction_logger(self):
        """Setup a dedicated logger for model predictions"""
        pred_logger = logging.getLogger("prediction_logger")
        pred_logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.predictions_log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        pred_logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        pred_logger.propagate = False
        
        return pred_logger

    def _collate_fn(self, batch):
        source_ids = torch.stack([item['source_text'] for item in batch])
        target_ids = torch.stack([item['target_text'] for item in batch])
        return {'source_text': source_ids, 'target_text': target_ids, 'raw_items': batch}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        valid_batches = 0
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / self.training_config['num_epochs']))
        
        # Log training epoch start
        self.prediction_logger.info(f"\n{'='*50}\nEPOCH {epoch} TRAINING\n{'='*50}")
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
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
                
                # Log sample predictions during training (every 50 batches)
                if batch_idx % 50 == 0:
                    self._log_predictions(batch, outputs, batch_idx, "train")
                
                if self.use_wandb:
                    wandb.log({'batch_loss': loss.item(), 'teacher_forcing_ratio': teacher_forcing_ratio})
            except Exception as e:
                logging.warning(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        valid_batches = 0
        
        # Log evaluation start
        self.prediction_logger.info(f"\n{'='*50}\nEVALUATION\n{'='*50}")
        
        for batch_idx, batch in enumerate(tqdm(self.valid_loader, desc="Evaluating")):
            try:
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                loss = self.criterion(outputs.view(-1, self.model_config['vocab_size']), target_ids.view(-1))
                
                total_loss += loss.item()
                valid_batches += 1
                
                predictions = outputs.argmax(dim=-1)
                
                # Log predictions for evaluation (more frequently)
                if batch_idx % 10 == 0:
                    self._log_predictions(batch, outputs, batch_idx, "eval")
                
                mask = target_ids != 0
                for pred, ref, m in zip(predictions, target_ids, mask):
                    valid_len = m.sum().item()
                    if valid_len > 0:
                        all_predictions.append(pred[:valid_len].cpu())
                        all_references.append(ref[:valid_len].cpu())
            except Exception as e:
                logging.warning(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
        
        if not all_predictions:
            logging.error("No valid predictions during evaluation")
            return {'val_loss': float('inf'), 'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0}
        
        metrics = self.metrics.compute_metrics(all_predictions, all_references)
        metrics['val_loss'] = total_loss / valid_batches if valid_batches > 0 else float('inf')
        
        # Log overall evaluation metrics
        self.prediction_logger.info(f"\nEvaluation Metrics: BLEU={metrics['bleu']:.4f}, ROUGE-L={metrics['rougeL']:.4f}, METEOR={metrics['meteor']:.4f}\n")
        
        return metrics

    def _log_predictions(self, batch, outputs, batch_idx, phase="train"):
        """Log model predictions with corresponding source and target texts"""
        predictions = outputs.argmax(dim=-1).cpu().numpy()
        source_ids = batch['source_text'].cpu().numpy()
        target_ids = batch['target_text'].cpu().numpy()
        
        # Sample up to 3 examples from the batch to log
        batch_size = len(predictions)
        num_samples = min(3, batch_size)
        sample_indices = random.sample(range(batch_size), num_samples)
        
        self.prediction_logger.info(f"\n{'-'*20} {phase.upper()} Batch {batch_idx} {'-'*20}")
        
        for idx in sample_indices:
            # Get non-padding tokens
            src_tokens = [t for t in source_ids[idx] if t != 0]
            tgt_tokens = [t for t in target_ids[idx] if t != 0]
            pred_tokens = [t for t in predictions[idx] if t != 0]
            
            # Decode tokens to text
            try:
                src_text = self.tokenizer.decode(src_tokens)
                tgt_text = self.tokenizer.decode(tgt_tokens)
                pred_text = self.tokenizer.decode(pred_tokens)
                
                self.prediction_logger.info(f"\nSample {idx}:")
                self.prediction_logger.info(f"SOURCE: {src_text}")
                self.prediction_logger.info(f"TARGET: {tgt_text}")
                self.prediction_logger.info(f"PREDICTION: {pred_text}")
                
                # Calculate BLEU score for this sample
                sample_bleu = self.metrics.compute_bleu_score([pred_tokens], [tgt_tokens])
                self.prediction_logger.info(f"Sample BLEU: {sample_bleu:.4f}")
                
            except Exception as e:
                self.prediction_logger.info(f"Error decoding sample {idx}: {str(e)}")

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = self.log_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            self.best_epoch = epoch
            best_path = self.log_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics
            }, best_path)
            logging.info(f"New best model saved with BLEU: {metrics['bleu']:.4f}")

    def generate_samples(self, num_samples=10):
        """Generate and log samples from validation set with beam search"""
        self.model.eval()
        
        # Log sample generation header
        self.prediction_logger.info(f"\n{'='*50}\nGENERATED SAMPLES\n{'='*50}")
        
        # Get random samples from validation set
        indices = random.sample(range(len(self.valid_dataset)), min(num_samples, len(self.valid_dataset)))
        
        for i, idx in enumerate(indices):
            sample = self.valid_dataset[idx]
            source_ids = torch.tensor(sample['source_text']).unsqueeze(0).to(self.device)
            target_ids = torch.tensor(sample['target_text']).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Generate with beam search (if implemented) or greedy
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1)
                
                # Decode tokens to text
                src_tokens = [t for t in source_ids[0].cpu().numpy() if t != 0]
                tgt_tokens = [t for t in target_ids[0].cpu().numpy() if t != 0]
                pred_tokens = [t for t in predictions[0].cpu().numpy() if t != 0]
                
                try:
                    src_text = self.tokenizer.decode(src_tokens)
                    tgt_text = self.tokenizer.decode(tgt_tokens)
                    pred_text = self.tokenizer.decode(pred_tokens)
                    
                    self.prediction_logger.info(f"\nGenerated Sample {i+1}:")
                    self.prediction_logger.info(f"SOURCE: {src_text}")
                    self.prediction_logger.info(f"TARGET: {tgt_text}")
                    self.prediction_logger.info(f"PREDICTION: {pred_text}")
                    
                    # Calculate BLEU score for this sample
                    sample_bleu = self.metrics.compute_bleu_score([pred_tokens], [tgt_tokens])
                    self.prediction_logger.info(f"Sample BLEU: {sample_bleu:.4f}")
                    
                    # Calculate longest common subsequence for ROUGE-L
                    sample_rouge = self.metrics.compute_rouge_l_score(pred_tokens, tgt_tokens)
                    self.prediction_logger.info(f"Sample ROUGE-L: {sample_rouge:.4f}")
                    
                except Exception as e:
                    self.prediction_logger.info(f"Error processing sample {i+1}: {str(e)}")

    def train(self):
        logging.info("Starting training")
        best_val_loss = float('inf')
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                # Train epoch
                train_loss = self.train_epoch(epoch)
                
                # Evaluate
                val_metrics = self.evaluate()
                
                # Update learning rate scheduler
                self.scheduler.step(val_metrics['val_loss'])
                
                # Log metrics
                logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, "
                             f"BLEU: {val_metrics['bleu']:.4f}, ROUGE-L: {val_metrics['rougeL']:.4f}, METEOR: {val_metrics['meteor']:.4f}")
                if self.use_wandb:
                    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr'], **val_metrics})
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics)
                
                # Generate and log samples every 2 epochs
                if epoch % 2 == 0 or epoch == self.training_config['num_epochs'] - 1:
                    self.generate_samples(num_samples=5)
                
                # Early stopping check
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.training_config.get('patience', 5):
                        logging.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
        except Exception as e:
            logging.error(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Generate final samples
            self.generate_samples(num_samples=10)
            
            # Final log messages
            if hasattr(self, 'best_epoch'):
                logging.info(f"Best model was from epoch {self.best_epoch} with BLEU: {self.best_bleu:.4f}")
            logging.info("Training completed")
            
            if self.use_wandb:
                wandb.finish()

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
