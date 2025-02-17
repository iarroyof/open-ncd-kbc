from src.trainers import AutoencoderTrainer
from src.data.tsv_text2text_dataset import ColumnConfig

if __name__ == "__main__":
    # Example configuration
    model_config = {
        'target_seq_len': 3,
        'd_model': 512,
        'hidden_dim': 256,
        'num_encoder_layers': 3,
        'dropout': 0.1,
        'activation': 'ReLU',
        'pe_mode': 'fixed',
        'use_normalization': True,
        'norm_type': 'batch',
        'max_seq_len': 5000
    }
    
    training_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'num_workers': 4,
        'chunk_size': 10000,
        'seed': 42
    }
    
    # Example data configuration for files without headers
    data_configs = [
        ColumnConfig(
            file_path="data/ncd_gp_conceptnet/ncd_gp_conceptnet_train.tsv",
            source_columns=[3, 2],  # First column contains source text
            target_columns=[4],  # Second column contains target text
            has_header=False,
            separator="\t"
        )
    ]
    
    # Initialize and run trainer
    trainer = AutoencoderTrainer(
        model_config=model_config,
        training_config=training_config,
        data_configs=data_configs,
        use_wandb=True  # Set to False if you don't want to use wandb
    )
    
    trainer.train()
