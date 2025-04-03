import logging
import argparse
from pathlib import Path
from typing import Dict
import wandb
import yaml
import os
import time
import ast

from src.trainers.base_trainer import BaseTrainer
from src.data.tsv_text2text_dataset import ColumnConfig
from src.prediction_logging import PredictionLogger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
import torch
import socket

workstation_name = socket.gethostname()
logging.info("Workstation Name: %s", workstation_name)
os.environ["WORKSTATION_NAME"] = workstation_name

# Global counter for round-robin GPU assignment
current_gpu_index = 0

def estimate_vram_usage(config: Dict) -> float:
    model_type = config['model_type']
    batch_size = config['batch_size']
    target_seq_len = config['target_seq_len']
    token_mem = 0.0001
    
    if model_type == 'autoencoder':
        d_model = config['autoencoder_d_model']
        num_layers = config['autoencoder_num_encoder_layers']
    elif model_type == 'attention_gru':
        d_model = config['attention_gru_embed_size']
        num_layers = config['attention_gru_num_layers']
    elif model_type == 'attention_lstm':
        d_model = config['attention_lstm_embed_size']
        num_layers = config['attention_lstm_num_layers']
    elif model_type == 'transformer':
        d_model = config['transformer_d_model']
        num_layers = config['transformer_num_encoder_layers'] + config['transformer_num_decoder_layers']
    elif model_type == 'conv_s2s':
        d_model = config['conv_s2s_embed_dim']
        num_layers = config['conv_s2s_num_layers']
    
    vram = batch_size * target_seq_len * d_model * num_layers * token_mem + 5
    return vram

def get_available_gpus() -> list:
    try:
        num_gpus = torch.cuda.device_count()
        available_gpus = list(range(num_gpus))
        logging.info(f"Detected {num_gpus} available GPUs in container: {available_gpus}")
        return available_gpus
    except Exception as e:
        logging.warning(f"GPU detection failed: {str(e)}. Defaulting to CPU.")
        return []

def assign_gpu(config: Dict, available_gpus: list, workstation_id: int) -> int:
    """Dynamically assign GPU from available list, respecting workstation mapping."""
    global current_gpu_index
    
    vram = estimate_vram_usage(config)
    logging.info(f"Estimated VRAM usage: {vram:.2f} GB")
    
    gpu_capacities = {0: 24, 1: 24, 2: 24, 3: 24, 4: 48, 5: 48}
    
    if workstation_id == 1:  # Santo
        base_gpus = [0, 1]
    elif workstation_id == 2:  # Blue-Demon
        base_gpus = [2, 3]
    elif workstation_id == 3:  # Lizmark
        base_gpus = [4, 5]
    else:
        raise ValueError(f"Invalid workstation_id: {workstation_id}")
    
    # Filter viable GPUs based on VRAM capacity
    viable_gpus = [gpu for gpu in available_gpus if base_gpus[gpu] in gpu_capacities and vram <= gpu_capacities[base_gpus[gpu]]]
    
    if not viable_gpus:
        logging.warning(f"No viable GPU for VRAM {vram:.2f} GB. Defaulting to first available.")
        gpu_id = base_gpus[0] if available_gpus else base_gpus[0]
    else:
        # Round-robin assignment across viable GPUs
        gpu_id = base_gpus[viable_gpus[current_gpu_index % len(viable_gpus)]]
        current_gpu_index += 1  # Increment for next run
    
    logging.info(f"Dynamically assigned GPU {gpu_id} from available GPUs")
    return gpu_id

def get_model_config(model_type: str, wandb_config: Dict = None) -> Dict:
    base_config = {
        'vocab_size': 32000,
        'target_seq_len': 64,
        'source_seq_len': 64,
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

    if wandb_config:
        prefix = f"{model_type}_"
        for key in config:
            wandb_key = prefix + key
            if wandb_key in wandb_config:
                config[key] = wandb_config[wandb_key]
        if 'target_seq_len' in wandb_config:
            config['target_seq_len'] = wandb_config['target_seq_len']
        if 'source_seq_len' in wandb_config:
            config['source_seq_len'] = wandb_config['source_seq_len']

    return config

def get_training_config(model_type: str, wandb_config: Dict = None) -> Dict:
    base_config = {
        'batch_size': 128,
        'num_epochs': 3,
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

    if wandb_config:
        for key in ['batch_size', 'learning_rate', 'num_epochs']:
            if key in wandb_config:
                config[key] = wandb_config[key]
    
    return config

def filter_wandb_config(full_config: Dict, model_type: str) -> Dict:
    """
    Returns a filtered dictionary containing only the hyperparameters relevant
    for the current model type plus generic ones (like data_path, batch_size, etc.).
    It also includes parameters that have been updated via the sweep.
    """
    filtered = {}
    # Generic keys to always log.
    generic_keys = ['model_type', 'data_path', 'batch_size', 'learning_rate',
                    'target_seq_len', 'source_seq_len', 'num_epochs',
                    'log_frequency', 'prediction_log_freq', 'prediction_samples', 'final_samples']
    for key in generic_keys:
        if key in full_config:
            filtered[key] = full_config[key]
    # Keys specific to the current model type are expected to be prefixed.
    prefix = f"{model_type}_"
    for key, value in full_config.items():
        if key.startswith(prefix):
            # Optionally, remove the prefix for logging clarity.
            new_key = key[len(prefix):]
            filtered[new_key] = value
    return filtered

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
    wandb_config_defaults = {
        'log_frequency': 100,
        'disable_code': True,
        'save_code': False,
        'log_model': False,
        'watch': False,
        'system_interval': 30,
    }
    os.environ["WANDB_SYSTEM_METRICS"] = "system.cpu,system.gpu.0.memory,system.gpu.1.memory,system.gpu.0.temp,system.gpu.1.temp,system.gpu.0.powerPercent,system.gpu.1.powerPercent,system.disk.free"
    with wandb.init() as run:
        config = wandb.config  # This is the full config from the sweep
        
        # First, determine current model type.
        model_type = config['model_type']
        
        # Filter configuration to include only hyperparameters relevant for the current model type.
        filtered_config = filter_wandb_config(dict(config), model_type)
        # Log this filtered configuration to wandb (this updates the run config visible in the UI)
        wandb.config.update(filtered_config, allow_val_change=True)
        
        workstation_name = os.environ.get('WORKSTATION_NAME', 'santo')
        cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
        
        if cuda_visible_devices is not None:
            logging.info(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
            gpu_id = cuda_visible_devices
        else:
            logging.info("CUDA_VISIBLE_DEVICES is not set. So it will be assigned automatically...")
            if workstation_name=='lizmark':
                workstation_id = 3
            elif workstation_name=='blue-demon':
                workstation_id = 2
            else:
                workstation_id = 1
            available_gpus = get_available_gpus()
            if not available_gpus:
                logging.error("No GPUs available in container. Exiting.")
                return
    
            gpu_id = assign_gpu(config, available_gpus, workstation_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        logging.info(f"Assigned run to GPU {gpu_id} (Workstation {workstation_name}) with VRAM {estimate_vram_usage(config):.2f} GB")
        
        train_path = config['data_path'][0]
        valid_path = config['data_path'][1]
        
        log_dir = f"logs/sweep_{run.id}"
        setup_logging(log_dir)
        logging.info(f"Starting sweep run {run.id} with model type: {model_type} on GPU {gpu_id}")
        
        # Get model and training configurations specific to the model type.
        model_config = get_model_config(model_type, config)
        training_config = get_training_config(model_type, config)
        
        # Log these configurations as part of the run configuration.
        wandb.log({'model_config': model_config, 'training_config': training_config})
        
        training_config['log_predictions'] = True
        training_config['prediction_log_freq'] = config.get('prediction_log_freq', 50)
        training_config['prediction_samples'] = config.get('prediction_samples', 3)
        
        train_configs, valid_configs = setup_data_configs(train_path, valid_path)
        
        trainer = BaseTrainer(
            model_type=model_type,
            model_config=model_config,
            training_config=training_config,
            train_configs=train_configs,
            valid_configs=valid_configs,
            log_dir=log_dir,
            use_wandb=True
        )        
        
        trainer.prediction_logger = PredictionLogger.setup_prediction_logger(
            Path(log_dir), 
            logger_name=f'predictions_{model_type}'
        )
        
        if not hasattr(trainer, '_log_predictions'):
            trainer._log_predictions = lambda batch, outputs, batch_idx, phase: \
                PredictionLogger.log_predictions(
                    trainer, batch, outputs, batch_idx, phase, 
                    max_samples=training_config['prediction_samples']
                )
        
        if not hasattr(trainer, 'generate_samples'):
            trainer.generate_samples = lambda num_samples=10: \
                PredictionLogger.generate_samples(trainer, num_samples)
        
        trainer.train()
        
        num_final_samples = config.get('final_samples', 10)
        trainer.generate_samples(num_samples=num_final_samples)
        
        del trainer
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Train sequence-to-sequence models with W&B sweeps in Docker')
    parser.add_argument('--model_type', type=str, default='autoencoder',
                        choices=['autoencoder', 'attention_gru', 'attention_lstm', 'transformer', 'conv_s2s'],
                        help='Type of model to train (ignored if using sweep)')
    parser.add_argument('--data_path', type=str,
                        help='Base path to data directory (ignored if using sweep)')
    parser.add_argument('--cache_dir', type=str, default='/app/cache',
                        help='Directory for caching datasets inside container')
    parser.add_argument('--log_dir', type=str, default='/app/logs',
                        help='Directory for logs and checkpoints inside container')
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
    parser.add_argument('--yaml', type=str, default=None,
                        help='Run W&B sweep yaml file')
    
    args, unknown = parser.parse_known_args()
    
    Path(args.cache_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.use_wandb:
            if not args.yaml:
                workstation_name = os.environ.get("WORKSTATION_NAME", "santo").lower().replace('-', '_')
                yaml_file = f'sweep_config_{workstation_name}.yaml'
            else:
                yaml_file = args.yaml
            with open(yaml_file, 'r') as f:
                sweep_config = yaml.safe_load(f)
            sweep_id = wandb.sweep(sweep_config, project="standard_models")
            for attempt in range(3):
                try:
                    wandb.agent(sweep_id, function=train_with_wandb, count=50)
                    break
                except Exception as e:
                    logging.error(f"Sweep attempt {attempt + 1} failed on {workstation_name}: {str(e)}. Retrying in 60 seconds...")
                    time.sleep(60)
        else:
            setup_logging(args.log_dir)
            logging.info(f"Starting script with model type: {args.model_type}")
            logging.info(f"Arguments: {vars(args)}")
            
            model_config = get_model_config(args.model_type)
            training_config = get_training_config(args.model_type)
            train_path, val_path = ast.literal_eval(args.data_path)
            train_configs, valid_configs = setup_data_configs(train_path, val_path)
            
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
