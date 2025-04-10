import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Adafactor
import logging
from pathlib import Path
from typing import List, Optional, Dict
import wandb
from tqdm import tqdm
import gc
import os
import random
import numpy as np
from torch.amp import GradScaler, autocast

# Assuming these imports are available in your project structure
from ..data.tsv_text2text_dataset import (
    CachedTSVDataset,
    ColumnConfig,
    CacheConfig,
    collate_fn
)
from ..metrics.evaluation import TextGenerationMetrics
from ..models.factory import build_model
from ..prediction_logging import PredictionLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BaseTrainer:
    def __init__(
        self,
        model_type: str,
        model_config: dict,
        training_config: dict,
        train_configs: List[ColumnConfig],
        valid_configs: List[ColumnConfig],
        tokenizer_path: Optional[str] = None,
        cache_dir: str = "./cache",
        log_dir: str = "./logs",
        use_wandb: bool = False
    ):
        self.model_type = model_type
        self.model_config = model_config.copy()
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 0

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=self.log_dir / "training.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        cache_config = CacheConfig(
            enable_cache=True,
            cache_dir=cache_dir,
            cache_format='h5'
        )

        self.train_dataset = CachedTSVDataset(
            configs=train_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('source_seq_len', 512)
        )

        self.valid_dataset = CachedTSVDataset(
            configs=valid_configs,
            cache_config=cache_config,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            max_length=self.model_config.get('source_seq_len', 512)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 2),
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 2),
            pin_memory=False,
            prefetch_factor=1,
            persistent_workers=False
        )
        
        self.model_config['vocab_size'] = self.train_dataset.get_vocab_size()
        self.model = build_model(model_type, self.model_config).to(self.device)
        pad_id = self.train_dataset.tokenizer.token_to_id("[PAD]")
        self.model.pad_id = pad_id
        self.model.sos_id = self.train_dataset.tokenizer.token_to_id("[BOS]")
        self.model.eos_id = self.train_dataset.tokenizer.token_to_id("[EOS]")
        
        self.scaler = GradScaler()

        if training_config.get("optimizer", "adafactor") == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(),
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config.get('weight_decay', 0.01)
            )
            
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=training_config.get('label_smoothing', 0.05)
        )

        self.metrics = TextGenerationMetrics(self.valid_dataset.tokenizer)
        self.use_wandb = use_wandb

    def trim_sequence_at_eos(self, seq, eos_token_id):
        """Trim a sequence at the EOS token."""
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        if eos_token_id in seq:
            return seq[:seq.index(eos_token_id) + 1]
        return seq

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.current_epoch = epoch
        self.model.train()
        total_loss, valid_batches = 0.0, 0
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(epoch)

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            try:
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                target_ids = self.pad_or_trim(target_ids)

                with autocast(device_type="cuda"):
                    outputs = self.model(src=source_ids, tgt=target_ids, teacher_forcing_ratio=teacher_forcing_ratio)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                valid_batches += 1

                if self.use_wandb:
                    wandb.log({"batch_loss": loss.item(), "teacher_forcing_ratio": teacher_forcing_ratio})

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise
        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    def get_teacher_forcing_ratio(self, epoch):
        """Calculate teacher forcing ratio based on epoch."""
        num_epochs = self.training_config['num_epochs']
        schedule = self.training_config.get('teacher_forcing_schedule', 'adaptive')

        if schedule == 'linear':
            return max(0.0, 1.0 - (epoch / num_epochs))
        elif schedule == 'adaptive':
            if num_epochs <= 5:
                return 0.5
            else:
                return max(0.0, 0.95 * (0.95 ** epoch))
        elif isinstance(schedule, float):
            return schedule
        else:
            return max(0.0, 1.0 - (epoch / num_epochs))

    def generate_seeded_samples(self, num_samples: int = 10, seed: int = 42) -> List[Dict[str, str]]:
        """
        Generate a specified number of prediction samples from the validation set using a fixed seed.
        
        Args:
            num_samples (int): Number of samples to generate
            seed (int): Random seed for reproducibility
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing source, target, and prediction text
        """
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.model.eval()
        sample_predictions = []
        eos_token_id = self.train_dataset.tokenizer.token_to_id("[EOS]")
        
        # Get total validation dataset size
        total_samples = len(self.valid_dataset)
        
        # Generate random indices to sample from
        if num_samples >= total_samples:
            # If requesting more samples than available, use all samples
            indices = list(range(total_samples))
            random.shuffle(indices)
            indices = indices[:num_samples]
        else:
            indices = random.sample(range(total_samples), num_samples)
        
        # Sort indices to access them in order for efficiency
        indices.sort()
        
        # Create a dictionary to keep track of which indices we still need to process
        remaining_indices = set(indices)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_loader):
                # Calculate which samples in this batch correspond to our target indices
                batch_size = len(batch['source_text'])
                batch_start_idx = batch_idx * self.training_config['batch_size']
                batch_end_idx = batch_start_idx + batch_size
                
                # Find which indices we need from this batch
                batch_indices_to_use = [
                    i - batch_start_idx for i in indices 
                    if i in remaining_indices and batch_start_idx <= i < batch_end_idx
                ]
                
                # If no indices in this batch match our target indices, skip to next batch
                if not batch_indices_to_use:
                    continue
                
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Only run model if we have indices to process
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                preds = outputs.argmax(dim=-1)
                
                for batch_idx in batch_indices_to_use:
                    global_idx = batch_start_idx + batch_idx
                    remaining_indices.remove(global_idx)
                    
                    src_trim = self.trim_sequence_at_eos(source_ids[batch_idx].cpu(), eos_token_id)
                    tgt_trim = self.trim_sequence_at_eos(target_ids[batch_idx].cpu(), eos_token_id)
                    pred_trim = self.trim_sequence_at_eos(preds[batch_idx].cpu(), eos_token_id)
                    
                    decoded_pred = self.train_dataset.tokenizer.decode(pred_trim)
                    decoded_target = self.train_dataset.tokenizer.decode(tgt_trim)
                    decoded_source = self.train_dataset.tokenizer.decode(src_trim)
                    
                    if not decoded_pred.strip():
                        decoded_pred = "[EMPTY]"
                    
                    # Compute metrics for this sample
                    metrics_result = self.metrics.compute_metrics(
                        [torch.tensor(pred_trim)],
                        [torch.tensor(tgt_trim)]
                    )
                    
                    sample_bleu = metrics_result.get("bleu", 0.0)
                    
                    sample_predictions.append({
                        'source': decoded_source.strip(),
                        'target': decoded_target.strip(),
                        'prediction': decoded_pred.strip(),
                        'bleu': sample_bleu
                    })
                
                # If we've found all the samples we need, break
                if not remaining_indices:
                    break
                
        return sample_predictions

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        torch.save(checkpoint, self.log_dir / f'checkpoint_epoch_{epoch}.pt')

    def evaluate(self, epoch: int = None) -> Dict[str, float]:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Evaluating"):
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                target_ids = self.pad_or_trim(target_ids)
                
                with autocast(device_type="cuda"):
                    outputs = self.model(src=source_ids, tgt=target_ids, teacher_forcing_ratio=0.0)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                
                total_loss += loss.item()
                
                # Process predictions for metrics
                preds = outputs.argmax(dim=-1).cpu()
                
                eos_token_id = self.train_dataset.tokenizer.token_to_id("[EOS]")
                
                for i in range(preds.size(0)):
                    pred_trim = self.trim_sequence_at_eos(preds[i], eos_token_id)
                    tgt_trim = self.trim_sequence_at_eos(target_ids[i].cpu(), eos_token_id)
                    
                    all_predictions.append(torch.tensor(pred_trim))
                    all_targets.append(torch.tensor(tgt_trim))
        
        # Calculate metrics
        metrics_results = self.metrics.compute_metrics(all_predictions, all_targets)
        
        # Add loss to metrics
        metrics_results['val_loss'] = total_loss / len(self.valid_loader)
        
        # Log metrics
        logging.info(f"Epoch {epoch if epoch is not None else 'N/A'} - Validation Metrics: {metrics_results}")
        
        if self.use_wandb:
            if epoch is not None:
                metrics_results['epoch'] = epoch
            wandb.log(metrics_results)
            
        return metrics_results

    def train(self, num_final_samples: int = 10, seed: int = 42):
        """
        Train the model and log predictions only at the final epoch.
        
        Args:
            num_final_samples (int): Number of samples to log at the final epoch
            seed (int): Random seed for sample selection
        """
        best_metrics = {'val_loss': float('inf'), 'bleu': 0.0}
        num_epochs = self.training_config['num_epochs']

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)
            
            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss})

            # Log predictions only at the final epoch
            if epoch == num_epochs - 1:
                final_samples = self.generate_seeded_samples(num_samples=num_final_samples, seed=seed)
                PredictionLogger.log_predictions(self, final_samples)

            # Save checkpoint if we have better validation metrics
            if val_metrics['val_loss'] < best_metrics['val_loss']:
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics)

    def pad_or_trim(self, target_ids):
        """Pad or trim target IDs to match target sequence length."""
        target_seq_len = self.model_config.get('target_seq_len', 512)
        if target_ids.size(1) > target_seq_len:
            return target_ids[:, :target_seq_len]
        elif target_ids.size(1) < target_seq_len:
            pad_len = target_seq_len - target_ids.size(1)
            return nn.functional.pad(target_ids, (0, pad_len), value=self.model.pad_id)
        return target_ids

    def __del__(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
