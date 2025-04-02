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
        sos_id = self.train_dataset.tokenizer.token_to_id("[SOS]")
        eos_id = self.train_dataset.tokenizer.token_to_id("[EOS]")
        print(f"[PAD]={pad_id},\n[SOS]={sos_id},\n[EOS]={eos_id}\n")
        self.model.pad_id = pad_id
        self.model.sos_id = sos_id
        self.model.eos_id = eos_id
        
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
            label_smoothing=training_config.get('label_smoothing', 0.1)
        )

        self.metrics = TextGenerationMetrics(self.train_dataset.tokenizer)

        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=f"{model_type}-trainer",
                config={
                    "model_config": self.model_config,
                    "training_config": training_config
                }
            )

    def trim_sequence_at_eos(self, seq, eos_token_id):
        if hasattr(seq, "tolist"):
            seq = seq.tolist()
        if eos_token_id in seq:
            return seq[: seq.index(eos_token_id) + 1]
        return seq

    def get_teacher_forcing_ratio(self, epoch):
        num_epochs = self.training_config['num_epochs']
        schedule = self.training_config.get('teacher_forcing_schedule', 'adaptive')

        if schedule == 'linear':
            return max(0.0, 1.0 - (epoch / num_epochs))
        elif schedule == 'adaptive':
            if num_epochs <= 3:
                return 0.5
            else:
                return max(0.0, 0.95 * (0.95 ** epoch))
        elif isinstance(schedule, float):
            return schedule
        else:
            return max(0.0, 1.0 - (epoch / num_epochs))

    def train_epoch(self, epoch: int) -> float:
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

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        sample_predictions = []
        eos_token_id = self.train_dataset.tokenizer.token_to_id("[EOS]")
        sos_token_id = self.train_dataset.tokenizer.token_to_id("[SOS]")

        SAMPLE_COUNT_PER_BATCH = 2
        MAX_SAMPLE_LOGS = 10

        for batch in tqdm(self.valid_loader, desc="Evaluating"):
            try:
                source_ids = batch['source_text'].to(self.device)
                target_ids = self.pad_or_trim(batch['target_text'].to(self.device))
                outputs = self.model(src=source_ids, teacher_forcing_ratio=0.0)
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = target_ids.reshape(-1)
                loss = self.criterion(outputs_flat, targets_flat)

                total_loss += loss.item()
                valid_batches += 1

                preds = outputs.argmax(dim=-1)

                if len(sample_predictions) < MAX_SAMPLE_LOGS:
                    for i in range(min(preds.size(0), SAMPLE_COUNT_PER_BATCH)):
                        src_trim = self.trim_sequence_at_eos(source_ids[i].cpu(), eos_token_id)
                        tgt_trim = self.trim_sequence_at_eos(target_ids[i].cpu(), eos_token_id)
                        pred_trim = self.trim_sequence_at_eos(preds[i].cpu(), eos_token_id)

                        decoded_pred = self.train_dataset.tokenizer.decode(pred_trim)
                        decoded_target = self.train_dataset.tokenizer.decode(tgt_trim)
                        decoded_source = self.train_dataset.tokenizer.decode(src_trim)

                        if not decoded_pred.strip():
                            decoded_pred = "[EMPTY] " + str(pred_trim[:5])

                        sample_predictions.append({
                            'source': decoded_source.strip(),
                            'target': decoded_target.strip(),
                            'prediction': decoded_pred.strip(),
                            'pred_tensor': pred_trim
                        })

                        logging.debug(f"Prediction tokens (first 5): {pred_trim[:5]}")

            except Exception as e:
                logging.warning(f"Error in evaluation batch: {str(e)}")
                continue

        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        trimmed_preds = [torch.tensor(self.train_dataset.tokenizer.encode(sp['prediction']).ids) for sp in sample_predictions]
        trimmed_refs = [torch.tensor(self.train_dataset.tokenizer.encode(sp['target']).ids) for sp in sample_predictions]

        try:
            metrics = self.metrics.compute_metrics(trimmed_preds, trimmed_refs)
        except Exception as e:
            logging.error(f"Error computing metrics: {str(e)}")
            metrics = {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0}

        metrics['val_loss'] = avg_loss
        PredictionLogger.log_evaluation_samples(self, sample_predictions)
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
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
        best_metrics = {'val_loss': float('inf'), 'bleu': 0.0}
        for epoch in range(self.training_config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            if self.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if val_metrics['val_loss'] < best_metrics['val_loss'] or val_metrics['bleu'] > best_metrics['bleu']:
                best_metrics = val_metrics
                self.save_checkpoint(epoch, val_metrics)

    def pad_or_trim(self, target_ids):
        if target_ids.size(1) > self.model_config['target_seq_len']:
            return target_ids[:, :self.model_config['target_seq_len']]
        elif target_ids.size(1) < self.model_config['target_seq_len']:
            pad_len = self.model_config['target_seq_len'] - target_ids.size(1)
            return nn.functional.pad(target_ids, (0, pad_len), value=0)
        return target_ids

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
