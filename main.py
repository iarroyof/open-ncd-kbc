# main.py

import torch
import logging
import argparse
from pathlib import Path
from typing import Dict
import wandb
import yaml

# Import all model trainers
from src.trainers.positional_autoencoder_trainer import AutoencoderTrainer
from src.trainers.attention_gru_trainer import AttentionGRUTrainer
from src.trainers.attention_lstm_trainer import AttentionLSTMTrainer
from src.trainers.transformer_trainer import TransformerTrainer
from src.trainers.conv_s2s_trainer import ConvS2STrainer
from src.data.tsv_text2text_dataset import ColumnConfig

def get_model_config(model_type: str, wandb_config: Dict = None) -> Dict:
    """Get model-specific configuration, optionally overridden by W&B config"""
    base_config = {
        'vocab_size': 32000,
        'target_seq_len': 64,
        'max_seq_len': 64,
        'dropout': 0.1
    }
    
    if model_type == 'autoencoder':
        config = {
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
        config = {
            **base_config,
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 2,
            'bidirectional_encoder': True,
            'use_attention': False
        }
    elif model_type == 'attention_lstm':
        config = {
            **base_config,
            'embed_size': 256,
            'hidden_size': 512,
            'num_layers': 2,
            'bidirectional_encoder': True,
            'dropout': 0.1,
            'use_attention': False
        }
    elif model_type == 'transformer':
        config = {
            **base_config,
            'd_model': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dim_feedforward': 2048,
            'activation': 'relu',
            'pe_mode': 'fixed',
            'fixed_scale': 1.0,
            'learned_scale': 1.0
        }
    elif model_type == 'conv_s2s':
        config = {
            **base_config,
            'embed_dim': 512,
            'hidden_dim': 512,
            'num_layers': 4,
            'kernel_size': 3,
            'dropout': 0.2,
            'use_attention': False
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Override with W&B config if provided
    if wandb_config:
        prefix = f"{model_type}_"
        for key in config:
            wandb_key = prefix + key
            if wandb_key in wandb_config:
                config[key] = wandb_config[wandb_key]
        # Common parameter
        if 'target_seq_len' in wandb_config:
            config['target_seq_len'] = wandb_config['target_seq_len']
    
    return config

def get_training_config(model_type: str, wandb_config: Dict = None) -> Dict:
    """Get training configuration, optionally overridden by W&B config"""
    base_config = {
        'batch_size': 128,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'num_workers': 4,
        'chunk_size': 10000,
        'seed': 42
    }
    
    if model_type == 'transformer':
        config = {
            **base_config,
            'learning_rate': 1e-4,
            'warmup_steps': 4000,
            'label_smoothing': 0.1
        }
    elif model_type == 'attention_gru':
        config = {
            **base_config,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0
        }
    elif model_type == 'attention_lstm':
        config = {
            **base_config,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5
        }
    elif model_type == 'conv_s2s':
        config = {
            **base_config,
            'learning_rate': 0.25,
            'weight_decay': 0.0,
            'label_smoothing': 0.1,
            'warmup_steps': 4000,
            'lr_decay': 0.1,
            'decay_steps': 50000
        }
    else:
        config = {
            **base_config,
            'learning_rate': 1e-3
        }

    # Override with W&B config if provided
    if wandb_config:
        for key in ['batch_size', 'learning_rate', 'num_epochs']:
            if key in wandb_config:
                config[key] = wandb_config[key]
    
    return config

def get_trainer_class(model_type: str):
    trainers = {
        'autoencoder': AutoencoderTrainer,
        'attention_gru': AttentionGRUTrainer,
        'attention_lstm': AttentionLSTMTrainer,
        'transformer': TransformerTrainer,
        'conv_s2s': ConvS2STrainer
    }
    if model_type not in trainers:
        raise ValueError(f"Unknown model type: {model_type}")
    return trainers[model_type]

def setup_data_configs(train_path: str, valid_path: str) -> tuple:
    """Setup data configurations with full paths"""
    train_configs = [
        ColumnConfig(
            file_path=train_path,
            source_columns=[3, 2],
            target_columns=[4],
            has_header=False,
            separator="\t",
            camel_to_lower=[2]
        )
    ]
    valid_configs = [
        ColumnConfig(
            file_path=valid_path,
            source_columns=[3, 2],
            target_columns=[4],
            has_header=False,
            separator="\t",
            camel_to_lower=[2]
        )
    ]
    return train_configs, valid_configs

def setup_logging(log_dir: str):
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "main.log"),
            logging.StreamHandler()
        ]
    )

def train_with_wandb():
    """Training function for W&B sweep"""
    with wandb.init() as run:
        config = wandb.config
        
        # Extract paths and model type from W&B config
        model_type = config['model_type']
        train_path = config['train_data_path']
        valid_path = config['valid_data_path']
        
        # Setup logging with W&B run ID
        setup_logging(f"logs/sweep_{run.id}")
        logging.info(f"Starting sweep run {run.id} with model type: {model_type}")
        
        # Get configurations from W&B
        model_config = get_model_config(model_type, config)
        training_config = get_training_config(model_type, config)
        
        # Setup data configurations
        train_configs, valid_configs = setup_data_configs(train_path, valid_path)
        
        # Initialize trainer
        TrainerClass = get_trainer_class(model_type)
        trainer = TrainerClass(
            model_config=model_config,
            training_config=training_config,
            train_configs=train_configs,
            valid_configs=valid_configs,
            tokenizer_path=None,
            cache_dir="./cache",
            log_dir=f"logs/sweep_{run.id}",
            use_wandb=True
        )
        
        # Train
        trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Train sequence-to-sequence models with W&B sweeps')
    parser.add_argument('--model_type', type=str, default='autoencoder',
                        choices=['autoencoder', 'attention_gru', 'attention_lstm', 'transformer', 'conv_s2s'],
                        help='Type of model to train (ignored if using sweep)')
    parser.add_argument('--data_path', type=str, default='data/ncd_gp_conceptnet',
                        help='Base path to data directory (ignored if using sweep)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching datasets')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to pretrained tokenizer (optional)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging and sweeps')
    parser.add_argument('--eval_only', action='store_true',
                        help='Run only evaluation on validation set')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint for evaluation')
    parser.add_argument('--sweep', action='store_true',
                        help='Run W&B sweep instead of single run')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.cache_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.use_wandb and args.sweep:
            # Setup W&B sweep
            with open('sweep_config.yaml', 'r') as f:
                sweep_config = yaml.safe_load(f)
            sweep_id = wandb.sweep(sweep_config, project="seq2seq_sweep")
            wandb.agent(sweep_id, function=train_with_wandb)
        else:
            # Single run mode
            setup_logging(args.log_dir)
            logging.info(f"Starting script with model type: {args.model_type}")
            logging.info(f"Arguments: {vars(args)}")
            
            # Get configurations
            model_config = get_model_config(args.model_type)
            training_config = get_training_config(args.model_type)
            
            # Setup data configurations with default paths
            train_configs = setup_data_configs(f"{args.data_path}/ncd_gp_conceptnet_train.tsv",
                                             f"{args.data_path}/ncd_gp_conceptnet_valid.tsv")[0]
            valid_configs = setup_data_configs(f"{args.data_path}/ncd_gp_conceptnet_train.tsv",
                                             f"{args.data_path}/ncd_gp_conceptnet_valid.tsv")[1]
            
            # Initialize trainer
            TrainerClass = get_trainer_class(args.model_type)
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
            
            if args.eval_only:
                logging.info("Running evaluation...")
                metrics = trainer.evaluate()
                for metric, value in metrics.items():
                    logging.info(f"{metric}: {value:.4f}")
            else:
                trainer.train()
                
    except Exception as e:
        logging.error(f"Process failed with error: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
