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
    wandb_config = {
        'log_frequency': 100,
        'disable_code': True,
        'save_code': False,
        'log_model': False,
        'watch': False,
        'system_interval': 30,
    }
    os.environ["WANDB_SYSTEM_METRICS"] = "system.cpu,system.gpu.0.memory,system.gpu.1.memory,system.gpu.0.temp,system.gpu.1.temp,system.gpu.0.powerPercent,system.gpu.1.powerPercent,system.disk.free"
    
    with wandb.init(config=None, settings=wandb.Settings(**wandb_config)) as run:
        config = wandb.config
        
        workstation_id = int(os.environ.get('WORKSTATION_ID', 1))
        workstation_name = os.environ.get('WORKSTATION_NAME', 'santo')
        
        available_gpus = get_available_gpus()
        if not available_gpus:
            logging.error("No GPUs available in container. Exiting.")
            return
        
        gpu_id = assign_gpu(config, available_gpus, workstation_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logging.info(f"Assigned run to GPU {gpu_id} (Workstation {workstation_name}) with VRAM {estimate_vram_usage(config):.2f} GB")
        
        model_type = config['model_type']
        train_path = config['train_data_path']
        valid_path = config['valid_data_path']
        
        log_dir = f"logs/sweep_{run.id}"
        setup_logging(log_dir)
        logging.info(f"Starting sweep run {run.id} with model type: {model_type} on GPU {gpu_id}")
        
        model_config = get_model_config(model_type, config)
        training_config = get_training_config(model_type, config)
        
        # Add prediction logging configuration
        training_config['log_predictions'] = True
        training_config['prediction_log_freq'] = config.get('prediction_log_freq', 50)  # Default to every 50 batches
        training_config['prediction_samples'] = config.get('prediction_samples', 3)     # Default to 3 samples per batch
        
        train_configs, valid_configs = setup_data_configs(train_path, valid_path)
        
        TrainerClass = get_trainer_class(model_type)
        trainer = TrainerClass(
            model_config=model_config,
            training_config=training_config,
            train_configs=train_configs,
            valid_configs=valid_configs,
            tokenizer_path=None,
            cache_dir="/app/cache",
            log_dir=log_dir,
            use_wandb=True
        )
        
        # Initialize prediction logger if the trainer class supports it
        if hasattr(trainer, '_setup_prediction_logger') and callable(getattr(trainer, '_setup_prediction_logger')):
            trainer.prediction_logger = trainer._setup_prediction_logger()
            
            # Set up WandB logging for predictions
            predictions_table = wandb.Table(columns=["epoch", "batch", "phase", "source", "target", "prediction", "bleu"])
            
            # Monkey patch the _log_predictions method if it exists
            if hasattr(trainer, '_log_predictions') and callable(getattr(trainer, '_log_predictions')):
                original_log_predictions = trainer._log_predictions
                
                def wandb_log_predictions(batch, outputs, batch_idx, phase="train"):
                    # Call the original method first
                    original_log_predictions(batch, outputs, batch_idx, phase)
                    
                    # Add to WandB table
                    predictions = outputs.argmax(dim=-1).cpu().numpy()
                    source_ids = batch['source_text'].cpu().numpy()
                    target_ids = batch['target_text'].cpu().numpy()
                    
                    # Sample up to n examples from the batch to log
                    batch_size = len(predictions)
                    num_samples = min(training_config['prediction_samples'], batch_size)
                    sample_indices = random.sample(range(batch_size), num_samples)
                    
                    current_epoch = getattr(trainer, 'current_epoch', 0)
                    
                    for idx in sample_indices:
                        try:
                            # Get non-padding tokens
                            src_tokens = [t for t in source_ids[idx] if t != 0]
                            tgt_tokens = [t for t in target_ids[idx] if t != 0]
                            pred_tokens = [t for t in predictions[idx] if t != 0]
                            
                            # Decode tokens to text
                            src_text = trainer.tokenizer.decode(src_tokens)
                            tgt_text = trainer.tokenizer.decode(tgt_tokens)
                            pred_text = trainer.tokenizer.decode(pred_tokens)
                            
                            # Calculate BLEU score for this sample
                            sample_bleu = trainer.metrics.compute_bleu_score([pred_tokens], [tgt_tokens])
                            
                            # Add to WandB table
                            predictions_table.add_data(
                                current_epoch, 
                                batch_idx, 
                                phase,
                                src_text, 
                                tgt_text, 
                                pred_text, 
                                sample_bleu
                            )
                        except Exception as e:
                            logging.warning(f"Error logging prediction to WandB: {str(e)}")
                
                # Replace the original method with our enhanced version
                trainer._log_predictions = wandb_log_predictions
            
            # Add a hook to log the table at the end of each epoch
            original_train_epoch = trainer.train_epoch
            
            def train_epoch_with_logging(epoch):
                # Store current epoch for logging
                trainer.current_epoch = epoch
                result = original_train_epoch(epoch)
                
                # Log predictions table to WandB at end of epoch
                wandb.log({"predictions": predictions_table})
                return result
            
            trainer.train_epoch = train_epoch_with_logging
                
        # Train the model
        trainer.train()
        
        # Log final predictions table if it exists
        if 'predictions_table' in locals():
            wandb.log({"final_predictions": predictions_table})
        
        # Generate samples and log them to WandB
        if hasattr(trainer, 'generate_samples') and callable(getattr(trainer, 'generate_samples')):
            num_final_samples = config.get('final_samples', 10)
            trainer.generate_samples(num_samples=num_final_samples)
            
            # If trainer has a prediction log file, upload it as an artifact
            if hasattr(trainer, 'predictions_log_path'):
                prediction_artifact = wandb.Artifact(
                    name=f"predictions_{run.id}", 
                    type="predictions", 
                    description="Model predictions log"
                )
                prediction_artifact.add_file(trainer.predictions_log_path)
                wandb.log_artifact(prediction_artifact)
        
        del trainer
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Train sequence-to-sequence models with W&B sweeps in Docker')
    parser.add_argument('--model_type', type=str, default='autoencoder',
                        choices=['autoencoder', 'attention_gru', 'attention_lstm', 'transformer', 'conv_s2s'],
                        help='Type of model to train (ignored if using sweep)')
    parser.add_argument('--data_path', type=str, default='data/ncd_gp_conceptnet',
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
    
    args, unknown = parser.parse_known_args()
    
    Path(args.cache_dir).mkdir(exist_ok=True)
    Path(args.log_dir).mkdir(exist_ok=True)
    
    try:
        if args.use_wandb and args.sweep:
            workstation_name = os.environ.get("WORKSTATION_NAME", "santo").lower().replace('-', '_')
            yaml_file = f'sweep_config_{workstation_name}.yaml'
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
            
            train_configs, valid_configs = setup_data_configs(
                f"{args.data_path}/ncd_gp_conceptnet_train.tsv",
                f"{args.data_path}/ncd_gp_conceptnet_valid.tsv"
            )
            
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
