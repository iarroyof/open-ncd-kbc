# prediction_logging.py
import logging
import torch
import random
import wandb
import pandas as pd

class PredictionLogger:
    @staticmethod
    def log_predictions(trainer, sample_predictions):
        """
        Log final epoch predictions from randomly sampled validation examples.
        
        Args:
            trainer: The trainer instance
            sample_predictions (list of dict): Each dict has 'source', 'target', and 'prediction' keys
        """
        if not sample_predictions:
            return
            
        # Set up prediction logger
        predictions_log_path = trainer.log_dir / "final_predictions.log"
        
        # Create predictions logger
        predictions_logger = logging.getLogger('final_predictions')
        predictions_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to prevent duplicate logs
        predictions_logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(predictions_log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to predictions logger
        predictions_logger.addHandler(file_handler)
        
        # Log predictions
        for i, sample in enumerate(sample_predictions):
            log_message = (
                f"Sample {i+1}: "
                f"Source: {sample['source']}, "
                f"Target: {sample['target']}, "
                f"Prediction: {sample['prediction']}"
            )
            predictions_logger.info(log_message)
            print(log_message)  # Also print to console
        
        # Log to wandb if enabled
        if trainer.use_wandb and wandb.run is not None:
            # Save samples in CSV format
            df = pd.DataFrame(sample_predictions)
            csv_path = predictions_log_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            # Create an artifact and log it
            prediction_artifact = wandb.Artifact(
                name="final_predictions", 
                type="predictions", 
                description="Final epoch sample predictions"
            )
            prediction_artifact.add_file(str(csv_path))
            wandb.log_artifact(prediction_artifact)
            
            # Also log as a table
            wandb.log({
                "final_predictions": wandb.Table(
                    dataframe=df
                )
            })
