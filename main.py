import logging
import argparse
from pathlib import Path
from typing import Dict
import wandb
import yaml
import os
import time
import ast
from functools import partial

from src.trainers.base_trainer import BaseTrainer
from src.data.tsv_text2text_dataset import ColumnConfig
from src.prediction_logging import PredictionLogger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import socket


import math

def estimate_memory_fraction(
    model_type: str,
    num_layers: int,
    hidden_size: int,
    seq_len_input: int,
    seq_len_output: int,
    batch_size: int,
    mixed_precision: bool = True,
    optimizer: str = "adafactor",
    gpu_total_memory_gb: float = 24.0,
    safety_buffer: float = 0.05
) -> float:
    dtype_size = 2 if mixed_precision else 4  # bytes per element
    vocab_size = 30000  # assume standard
    num_heads = max(1, hidden_size // 64)

    # Estimate params: encoder + decoder layers
    if model_type.lower() == "transformer":
        params = 12 * hidden_size**2 * num_layers
    elif model_type.lower() in {"attention_gru", "attention_lstm"}:
        params = 6 * hidden_size**2 * num_layers
    elif model_type.lower() in ['autoencoder', 'conv_s2s']:
        params = 4 * hidden_size**2 * num_layers
    else:
        logging.error(f"Model type {model_type} is not implemented, so we are going to terminate...")
        exit()

    param_mem = params * dtype_size / 1e9  # in GB

    # Activation memory (forward + backward)
    act_tokens = batch_size * (seq_len_input + seq_len_output)
    act_mem = act_tokens * hidden_size * num_layers * dtype_size * 2 / 1e9  # in GB

    # Optimizer memory
    if optimizer.lower() == "adafactor":
        opt_mem = 0.5 * param_mem
    else:  # e.g., Adam
        opt_mem = 2 * param_mem

    # Total estimated
    total_mem = param_mem + act_mem + opt_mem

    # Leave headroom
    total_mem *= (1 + safety_buffer)

    return min(round(total_mem / gpu_total_memory_gb, 2), 0.98)  # Cap at 98%

def get_memory_estimate_kwargs(config, total_vram=24, safety_buff=0.05):
    model_type = config["model_type"].lower()
    base_kwargs = {
        "model_type": model_type,
        "seq_len_input": config["source_seq_len"],
        "seq_len_output": config["target_seq_len"],
        "batch_size": config["batch_size"],
        "mixed_precision": config["mixed_precision"] if "mixed_precision" in config else True,
        "optimizer": config["optimizer"],
        "gpu_total_memory_gb": total_vram,
        "safety_buffer": safety_buff,
    }

    # Map per-model architecture keys
    if model_type == "transformer":
        base_kwargs.update({
            "hidden_size": config["transformer_d_model"],
            "num_layers": config["transformer_num_encoder_layers"] + config["transformer_num_decoder_layers"],
        })
    elif model_type == "attention_gru":
        base_kwargs.update({
            "hidden_size": config["attention_gru_hidden_size"],
            "num_layers": config["attention_gru_num_layers"],
        })
    elif model_type == "attention_lstm":
        base_kwargs.update({
            "hidden_size": config["attention_lstm_hidden_size"],
            "num_layers": config["attention_lstm_num_layers"],
        })
    elif model_type == "conv_s2s":
        base_kwargs.update({
            "hidden_size": config["conv_s2s_hidden_dim"],
            "num_layers": config["conv_s2s_num_layers"],
        })
    elif model_type == "autoencoder":
        base_kwargs.update({
            "hidden_size": config["autoencoder_hidden_dim"],
            "num_layers": config["autoencoder_num_encoder_layers"],  # Assuming symmetrical AE
        })
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return base_kwargs
    
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
        'seed': 42,
        'optimizer': 'adafactor'
    }
    if model_type == 'transformer':
        config = {
            **base_config,
            'learning_rate': None if base_config['optimizer'] == 'adafactor' else 1e-4,
            'warmup_steps': 4000,
            'label_smoothing': 0.1
        }
    elif model_type == 'attention_gru':
        config = {
            **base_config,
            'learning_rate': None if base_config['optimizer'] == 'adafactor' else 1e-3,
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
            'learning_rate': None if base_config['optimizer'] == 'adafactor' else 0.25,
            'weight_decay': 0.0,
            'label_smoothing': 0.1,
            'warmup_steps': 4000,
            'lr_decay': 0.1,
            'decay_steps': 50000
        }
    else:
        config = {
            **base_config,
            'learning_rate': None if base_config['optimizer'] == 'adafactor' else 1e-3
        }
    if wandb_config:
        for key in ['batch_size', 'learning_rate', 'num_epochs']:
            if key in wandb_config:
                config[key] = wandb_config[key]
    return config

def filter_wandb_config(full_config: Dict, model_type: str) -> Dict:
    filtered = {}
    generic_keys = ['model_type', 'data_path', 'batch_size', 'learning_rate',
                    'target_seq_len', 'source_seq_len', 'num_epochs',
                    'log_frequency', 'prediction_log_freq', 'prediction_samples', 'final_samples']
    for key in generic_keys:
        if key in full_config:
            filtered[key] = full_config[key]
    prefix = f"{model_type}_"
    for key, value in full_config.items():
        if key.startswith(prefix):
            filtered[key[len(prefix):]] = value
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
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "main.log"),
            logging.StreamHandler()
        ]
    )

def train_with_wandb(run_config: Dict):
    """
    The run_config dictionary includes:
      - parent_log_dir: Path object for the parent log directory.
      - debug_log_predictions: Boolean flag.
      - workstation_name: The workstation name.
    """
    os.environ["WANDB_SYSTEM_METRICS"] = (
        "system.cpu,system.gpu.0.memory,system.gpu.1.memory,"
        "system.gpu.0.temp,system.gpu.1.temp,"
        "system.gpu.0.powerPercent,system.gpu.1.powerPercent,"
        "system.disk.free"
    )
    with wandb.init() as run:
        config = wandb.config
        model_type = config['model_type']
        filtered_config = filter_wandb_config(dict(config), model_type)
        wandb.config.update(filtered_config, allow_val_change=True)
        if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(current_device).total_memory
                total_memory_gb = total_memory / (1024 ** 3)
                memory_kwargs = get_memory_estimate_kwargs(wandb.config, total_vram=total_memory_gb, safety_buff=0.05)
                fraction = estimate_memory_fraction(**memory_kwargs)
                torch.cuda.set_per_process_memory_fraction(fraction, device=current_device)                
                logging.info(f"The current CUDA device index: {current_device} was assigned {fraction}% GPU memory to the current process.")
        else:
                print("CUDA not available.")
        
        logging.info(f"Using workstation: {run_config['workstation_name']}")
        train_path = config['data_path'][0]
        valid_path = config['data_path'][1]
        
        parent_log_dir = run_config["parent_log_dir"]
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id is not None else "default_sweep"
        log_dir = parent_log_dir / sweep_id / wandb.run.id
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(str(log_dir))
        logging.info(f"Starting sweep run {wandb.run.id} (sweep: {sweep_id}) with model type: {model_type}")
        
        model_config = get_model_config(model_type, config)
        training_config = get_training_config(model_type, config)
        #wandb.log({'model_config': model_config, 'training_config': training_config})
        
        training_config['prediction_samples'] = config.get('prediction_samples', 3)
        if run_config["debug_log_predictions"]:
            training_config['log_predictions'] = True
            training_config['prediction_log_freq'] = config.get('prediction_log_freq', 50)
        else:
            training_config['log_predictions'] = False
            training_config['prediction_log_freq'] = None
        
        train_configs, valid_configs = setup_data_configs(train_path, valid_path)
        
        trainer = BaseTrainer(
            model_type=model_type,
            model_config=model_config,
            training_config=training_config,
            train_configs=train_configs,
            valid_configs=valid_configs,
            log_dir=str(log_dir),
            use_wandb=True
        )
        trainer.train()
        
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
    parser.add_argument('--project', type=str, default="standard_models",
                        help='W&B project name')
    parser.add_argument('--yaml', type=str, default=None,
                        help='W&B sweep yaml file')
    parser.add_argument('--debug_log_predictions', action='store_true',
                        help='Enable frequent prediction logging for debugging purposes')
    
    args, unknown = parser.parse_known_args()
    workstation_name = os.environ.get("WORKSTATION_NAME", socket.gethostname())
    logging.info("Workstation Name: %s", workstation_name)
    run_config = {
        "parent_log_dir": Path(args.log_dir),
        "debug_log_predictions": args.debug_log_predictions,
        "workstation_name": workstation_name
    }
    
    Path(args.cache_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.use_wandb:
            if not args.yaml:
                yaml_file = f'sweep_config_{run_config["workstation_name"]}.yaml'
            else:
                yaml_file = args.yaml
            with open(yaml_file, 'r') as f:
                sweep_config = yaml.safe_load(f)
            sweep_id = wandb.sweep(sweep_config, project=args.project)
            agent_fn = partial(train_with_wandb, run_config)

            for attempt in range(3):
                try:
                    wandb.agent(sweep_id, function=agent_fn, count=50)
                    break
                except Exception as e:
                    logging.error(f"Sweep attempt {attempt + 1} failed on {run_config['workstation_name']}: {str(e)}. Retrying in 60 seconds...")
                    time.sleep(60)
        else:
            setup_logging(args.log_dir)
            logging.info(f"Starting script with model type: {args.model_type}")
            logging.info(f"Arguments: {vars(args)}")
            
            model_config = get_model_config(args.model_type)
            training_config = get_training_config(args.model_type)
            if args.debug_log_predictions:
                training_config['log_predictions'] = True
                training_config['prediction_log_freq'] = 50
            else:
                training_config['log_predictions'] = False
                training_config['prediction_log_freq'] = None
            training_config['prediction_samples'] = training_config.get('prediction_samples', 3)
            
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
