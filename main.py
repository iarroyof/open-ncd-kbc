import torch
import logging
import argparse
from pathlib import Path
from typing import Dict
import wandb
import yaml
import os
import time

from src.trainers.positional_autoencoder_trainer import AutoencoderTrainer
from src.trainers.attention_gru_trainer import AttentionGRUTrainer
from src.trainers.attention_lstm_trainer import AttentionLSTMTrainer
from src.trainers.transformer_trainer import TransformerTrainer
from src.trainers.conv_s2s_trainer import ConvS2STrainer
from src.data.tsv_text2text_dataset import ColumnConfig

def estimate_vram_usage(config: Dict) -> float:
    """Estimate VRAM usage in GB based on hyperparameters."""
    model_type = config['model_type']
    batch_size = config['batch_size']
    target_seq_len = config['target_seq_len']
    token_mem = 0.0001  # GB per token per batch item per dim
    
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
    
    vram = batch_size * target_seq_len * d_model * num_layers * token_mem + 5  # 5 GB baseline overhead
    return vram

def get_available_gpus() -> list:
    """Detect available GPUs within the container."""
    try:
        num_gpus = torch.cuda.device_count()
        available_gpus = list(range(num_gpus))
        logging.info(f"Detected {num_gpus} available GPUs in container: {available_gpus}")
        return available_gpus
    except Exception as e:
        logging.warning(f"GPU detection failed: {str(e)}. Defaulting to CPU.")
        return []

def assign_gpu(config: Dict, available_gpus: list, workstation_id: int) -> int:
    """Assign GPU based on VRAM usage and workstation-specific mapping."""
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
    
    viable_gpus = [gpu for gpu in available_gpus if base_gpus[gpu] in gpu_capacities and vram <= gpu_capacities[base_gpus[gpu]]]
    
    if not viable_gpus:
        logging.warning(f"No viable GPU for VRAM {vram:.2f} GB. Defaulting to first available.")
        return base_gpus[0] if available_gpus else base_gpus[0]
    
    local_gpu = viable_gpus[hash(str(config)) % len(viable_gpus)]
    return base_gpus[local_gpu]

def get_model_config(model_type: str, wandb_config: Dict = None) -> Dict:
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

    if wandb_config:
        prefix = f"{model_type}_"
        for key in config:
            wandb_key = prefix + key
            if wandb_key in wandb_config:
                config[key] = wandb_config[wandb_key]
        if 'target_seq_len' in wandb_config:
            config['target_seq_len'] = wandb_config['target_seq_len']
    
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
    valid_map = {
        'data/conceptnet_gp/conceptnet_gp_train.tsv': 'data/conceptnet_gp/conceptnet_gp_valid.tsv',
        'data/ncd_gp_conceptnet/ncd_gp_conceptnet_train.tsv': 'data/ncd_gp_conceptnet/ncd_gp_conceptnet_valid.tsv'
    }
    if valid_map[train_path] != valid_path:
        raise ValueError(f"Dataset mismatch: {train_path} requires {valid_map[train_path]}, got {valid_path}")
    
    train_configs = [
        ColumnConfig(
            file_path=train
