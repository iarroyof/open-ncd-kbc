# main.py
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict

# Import all model trainers
from src.trainers.positional_autoencoder_trainer import AutoencoderTrainer
from src.trainers.attention_gru_trainer import AttentionGRUTrainer
from src.data.tsv_text2text_dataset import ColumnConfig

def get_model_config(model_type: str) -> Dict:
    """Get model-specific configuration"""
    base_config = {
        'vocab_size': 32000,
        'target_seq_len': 64,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    if model_type == 'autoencoder':
        return {
            **base_config,
            'd_model': 2048,
            'hidden_dim': 1024,
            'num_encoder_layers': 2,
            'activation': 'ReLU',
            'pe_mode': 'fixed',
            'use_normalization': True,
            'norm_type': 'batch'
        }
    
    elif model_type == 'attention_gru':
        return {
            **base_config,
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 2,
            'bidirectional_encoder': True
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_training_config(model_type: str) -> Dict:
    """Get training configuration, can be model-specific if needed"""
    return {
        'batch_size': 128,
        'learning_rate': 1e-3,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'num_workers': 4,
        'chunk_size': 10000,
        'seed': 42
    }

def get_trainer_class(model_type: str):
    """Get the appropriate trainer class"""
    trainers = {
        'autoencoder': AutoencoderTrainer,
        'attention_gru': AttentionGRUTrainer
    }
    
    if model_type not in trainers:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return trainers[model_type]

def setup_data_configs(data_path: str, split: str = 'train') -> list:
    """Setup data configurations"""
    return [
        ColumnConfig(
            file_path=f"{data_path}/ncd_gp_conceptnet_{split}.tsv",
            source_columns=[3, 2],  # First column contains source text
            target_columns=[4],     # Second column contains target text
            has_header=False,
            separator="\t"
        )
    ]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train various sequence-to-sequence models')
    parser.add_argument('--model_type', type=str, default='autoencoder',
                      choices=['autoencoder', 'attention_gru'],
                      help='Type of model to train')
    parser.add_argument('--data_path', type=str, default='data/ncd_gp_conceptnet',
                      help='Path to data directory')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                      help='Directory for caching datasets')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory for logs and checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                      help='Path to pretrained tokenizer (optional)')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Whether to use Weights & Biases logging')
    parser.add_argument('--eval_only', action='store_true',
                      help='Run only evaluation on validation set')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                      help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    Path(args.cache_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)
    
    # Get configurations
    model_config = get_model_config(args.model_type)
    training_config = get_training_config(args.model_type)
    
    # Setup data configurations
    train_configs = setup_data_configs(args.data_path, 'train')
    valid_configs = setup_data_configs(args.data_path, 'valid')
    
    # Get appropriate trainer class
    TrainerClass = get_trainer_class(args.model_type)
    
    try:
        # Initialize trainer
        trainer = TrainerClass(
            model_config=model_config,
            training_config=training_config,
            train_configs=train_configs,
            valid_configs=valid_configs,
            tokenizer_path=args.tokenizer_path,
            cache_dir=args.cache_dir,
            log_dir=args.log_dir,
            use_wandb=args.use_wandb
        )
        
        # Load checkpoint if provided
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            if not args.eval_only:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logging.info(f"Resuming training from epoch {checkpoint['epoch']}")
        
        if args.eval_only:
            # Run evaluation only
            logging.info("Running evaluation...")
            metrics = trainer.evaluate()
            for metric, value in metrics.items():
                logging.info(f"{metric}: {value:.4f}")
        else:
            # Run training
            trainer.train()
            
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
    except Exception as e:
        logging.error(f"Process failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
