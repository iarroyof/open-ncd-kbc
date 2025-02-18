# main.py
from src.trainers.positional_autoencoder_trainer import AutoencoderTrainer
from src.data.tsv_text2text_dataset import ColumnConfig, CacheConfig
import logging
import os
from pathlib import Path

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    Path("./cache").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    # Model configuration
    model_config = {
        'vocab_size': 32000,
        'target_seq_len': 64,
        'd_model': 2048,
        'hidden_dim': 1024,
        'num_encoder_layers': 2,
        'dropout': 0.1,
        'activation': 'ReLU',
        'pe_mode': 'fixed',
        'use_normalization': True,
        'norm_type': 'batch',
        'max_seq_len': 30
    }
    
    # Training configuration
    training_config = {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'num_workers': 4,
        'chunk_size': 10000,
        'seed': 42
    }
    
    # Training data configuration
    train_configs = [
        ColumnConfig(
            file_path="data/ncd_gp_conceptnet/ncd_gp_conceptnet_train.tsv",
            source_columns=[3, 2],  # First column contains source text
            target_columns=[4],     # Second column contains target text
            has_header=False,
            separator="\t"
        )
    ]
    
    # Validation data configuration
    valid_configs = [
        ColumnConfig(
            file_path="data/ncd_gp_conceptnet/ncd_gp_conceptnet_valid.tsv",
            source_columns=[3, 2],  # Same column structure as training data
            target_columns=[4],
            has_header=False,
            separator="\t"
        )
    ]
    
    # Initialize and run trainer
    trainer = AutoencoderTrainer(
        model_config=model_config,
        training_config=training_config,
        train_configs=train_configs,    # Training data configuration
        valid_configs=valid_configs,    # Validation data configuration
        cache_dir="./cache",           # Directory for cached datasets
        log_dir="./logs",             # Directory for logging and checkpoints
        use_wandb=True                # Set to False if you don't want to use wandb
    )
    
    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
