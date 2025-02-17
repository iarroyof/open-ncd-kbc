import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import List, Optional
import wandb
from tqdm import tqdm
from ..data.tsv_text2text_dataset import *
from ..models.text2text_autoencoders import PositionalAutoencoder
import os

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AutoencoderTrainer:
    def __init__(
        self,
        model_config: dict,
        training_config: dict,
        data_configs: List[ColumnConfig],
        tokenizer_path: Optional[str] = None,
        cache_dir: str = "./cache",
        log_dir: str = "./logs",
        use_wandb: bool = False
    ):
        self.model_config = model_config.copy()  # Make a copy to avoid modifying original
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
        
        # Initialize dataset with vocab_size from model_config
        self.dataset = TSVText2TextDataset(
            configs=data_configs,
            tokenizer_path=tokenizer_path,
            vocab_size=self.model_config.get('vocab_size', 32000),
            cache_dir=cache_dir,
            seed=training_config.get('seed', 42),
            chunk_size=training_config.get('chunk_size', 10000)
        )
        
        # Initialize dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=training_config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Update model_config with vocab_size from dataset
        self.model_config['vocab_size'] = self.dataset.get_vocab_size()
        
        # Initialize model with updated config
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
            steps_per_epoch=len(self.dataloader),
            pct_start=0.1  # 10% warmup
        )
        
        # Initialize wandb if requested
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
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move data to device
                source_ids = batch['source_text'].to(self.device)
                target_ids = batch['target_text'].to(self.device)
                
                # Ensure target sequence length matches model's expected length
                if target_ids.size(1) < self.model.target_seq_len:
                    # If target is too short, pad it
                    pad_length = self.model.target_seq_len - target_ids.size(1)
                    target_ids = torch.nn.functional.pad(target_ids, (0, pad_length), value=0)
                else:
                    # If target is too long, truncate it
                    target_ids = target_ids[:, :self.model.target_seq_len]
                
                # Forward pass
                outputs = self.model(source_ids)
                
            # Calculate loss
            batch_size, seq_len, vocab_size = outputs.shape
            # Ensure target_ids is the right shape and length
            target_ids = target_ids[:, :seq_len].contiguous()
            # Reshape for loss calculation
            outputs = outputs.view(-1, vocab_size)
            target_ids = target_ids.view(-1)
            # Calculate loss
            loss = nn.CrossEntropyLoss()(outputs, target_ids)
            
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
        
        epoch_loss = total_loss / len(self.dataloader)
        return epoch_loss

    def save_checkpoint(self, epoch: int, loss: float):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        torch.save(checkpoint, self.log_dir / f'checkpoint_epoch_{epoch}.pt')

    def train(self):
        logging.info("Starting training")
        best_loss = float('inf')
        
        for epoch in range(self.training_config['num_epochs']):
            epoch_loss = self.train_epoch(epoch)
            
            # Log metrics
            logging.info(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch_loss': epoch_loss
                })
            
            # Save checkpoint if best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(epoch, epoch_loss)
                logging.info(f"Saved new best model with loss: {epoch_loss:.4f}")

        logging.info("Training completed")
        if self.use_wandb:
            wandb.finish()
