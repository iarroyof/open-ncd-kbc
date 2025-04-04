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
from torch.amp import GradScaler, autocast

# Assuming these imports are available in your project structure
from ..data.tsv_text2text_dataset import (
    CachedTSVDataset,
    ColumnConfig,
    CacheConfig,
    collate_fn
)
from ..metrics.evaluation import TextGenerationMetrics
from ..prediction_logging import PredictionLogger
from ..models.factory import build_model

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
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=False
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 2),
            pin_memory=True,
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

        self.metrics = TextGenerationMetrics(self.train_dataset.tokenizer)
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

    def generate_samples(self, num_samples: int = 10) -> List[Dict[str, str]]:
        """Generate a specified number of prediction samples from the validation set."""
        self.model.eval()
        sample_predictions = []
        eos_token_id = self.train_dataset.tokenizer.token_to_id("[EOS]")

        with torch.no_grad():
            for batch in self.valid_loader:
                if len(sample_predictions) >= num_samples:
                    break
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                preds = outputs.argmax(dim=-1)

                for i in range(preds.size(0)):
                    if len(sample_predictions) >= num_samples:
                        break
                    src_trim = self.trim_sequence_at_eos(source_ids[i].cpu(), eos_token_id)
                    tgt_trim = self.trim_sequence_at_eos(target_ids[i].cpu(), eos_token_id)
                    pred_trim = self.trim_sequence_at_eos(preds[i].cpu(), eos_token_id)

                    decoded_pred = self.train_dataset.tokenizer.decode(pred_trim)
                    decoded_target = self.train_dataset.tokenizer.decode(tgt_trim)
                    decoded_source = self.train_dataset.tokenizer.decode(src_trim)

                    if not decoded_pred.strip():
                        decoded_pred = "[EMPTY]"

                    sample_predictions.append({
                        'source': decoded_source.strip(),
                        'target': decoded_target.strip(),
                        'prediction': decoded_pred.strip()
                    })

                    # Optional: Log to stdout
                    print(f"Sample {len(sample_predictions)} - Source: {decoded_source.strip()}")
                    print(f"Target: {decoded_target.strip()}")
                    print(f"Prediction: {decoded_pred.strip()}\n")

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

    def train(self):
        """Train the model and log predictions only at the final epoch."""
        best_metrics = {'val_loss': float('inf'), 'bleu': 0.0}
        num_epochs = self.training_config['num_epochs']

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            # Optional evaluation during training (not required for final predictions)
            # val_metrics = self.evaluate(epoch)
            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss})

            # Log predictions only at the final epoch
            if epoch == num_epochs - 1:
                final_samples = self.generate_samples(num_samples=10)
                PredictionLogger.log_predictions(self, final_samples)
                if self.use_wandb:
                    wandb.log({"final_predictions": final_samples})

            # Optional checkpointing (uncomment if needed)
            # if val_metrics['val_loss'] < best_metrics['val_loss']:
            #     best_metrics = val_metrics
            #     self.save_checkpoint(epoch, val_metrics)

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
