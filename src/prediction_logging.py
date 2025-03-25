# prediction_logging.py
import logging
import torch
import random
import wandb

class PredictionLogger:
    @staticmethod
    def setup_prediction_logger(log_dir, logger_name='predictions'):
        """
        Set up a dedicated logger for predictions.
        
        Args:
            log_dir (Path): Directory to store prediction logs
            logger_name (str, optional): Name of the logger
        
        Returns:
            logging.Logger: Configured prediction logger
        """
        predictions_log_path = log_dir / f"{logger_name}.log"
        
        # Create predictions logger
        predictions_logger = logging.getLogger(logger_name)
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
        
        return predictions_logger

    @staticmethod
    def log_predictions(
        trainer, 
        batch, 
        outputs, 
        batch_idx, 
        phase="train", 
        max_samples=3
    ):
        """
        Log model predictions for a batch.
        
        Args:
            trainer: The trainer instance
            batch (dict): Input batch containing source and target texts
            outputs (torch.Tensor): Model predictions
            batch_idx (int): Current batch index
            phase (str, optional): Training phase. Defaults to "train".
            max_samples (int, optional): Maximum number of samples to log
        """
        # Ensure prediction logger and tokenizer exist
        if not hasattr(trainer, 'prediction_logger') or \
           not hasattr(trainer, 'train_dataset') or \
           not hasattr(trainer.train_dataset, 'tokenizer'):
            return
        
        try:
            # Convert outputs to token indices
            predictions = outputs.argmax(dim=-1).cpu().numpy()
            source_ids = batch['source_text'].cpu().numpy()
            target_ids = batch['target_text'].cpu().numpy()
            
            # Sample up to max_samples examples from the batch
            batch_size = len(predictions)
            num_samples = min(max_samples, batch_size)
            sample_indices = random.sample(range(batch_size), num_samples)
            
            current_epoch = getattr(trainer, 'current_epoch', 0)
            
            for idx in sample_indices:
                # Decode non-padding tokens
                src_tokens = [t for t in source_ids[idx] if t != 0]
                tgt_tokens = [t for t in target_ids[idx] if t != 0]
                pred_tokens = [t for t in predictions[idx] if t != 0]
                
                # Decode tokens to text
                src_text = trainer.train_dataset.tokenizer.decode(src_tokens)
                tgt_text = trainer.train_dataset.tokenizer.decode(tgt_tokens)
                pred_text = trainer.train_dataset.tokenizer.decode(pred_tokens)
                
                # Calculate BLEU score for this sample
                sample_bleu = trainer.metrics.compute_bleu_score([pred_tokens], [tgt_tokens])
                
                # Log prediction details
                log_message = (
                    f"Epoch: {current_epoch}, "
                    f"Phase: {phase}, "
                    f"Batch: {batch_idx}, "
                    f"Source: {src_text}, "
                    f"Target: {tgt_text}, "
                    f"Prediction: {pred_text}, "
                    f"BLEU: {sample_bleu:.4f}"
                )
                trainer.prediction_logger.info(log_message)
                
                # Optional: Log to wandb if available
                if trainer.use_wandb:
                    wandb.log({
                        f"{phase}_predictions": wandb.Table(
                            columns=["Epoch", "Phase", "Batch", "Source", "Target", "Prediction", "BLEU"],
                            data=[[
                                current_epoch, 
                                phase, 
                                batch_idx, 
                                src_text, 
                                tgt_text, 
                                pred_text, 
                                sample_bleu
                            ]]
                        )
                    })
        except Exception as e:
            logging.warning(f"Error logging predictions: {str(e)}")

    @staticmethod
    def generate_samples(
        trainer, 
        num_samples=10, 
        samples_per_batch=3
    ):
        """
        Generate and log prediction samples from the validation dataset.
        
        Args:
            trainer: The trainer instance
            num_samples (int, optional): Total number of samples to generate
            samples_per_batch (int, optional): Max samples to log per batch
        """
        trainer.model.eval()
        samples_log_path = trainer.log_dir / "generated_samples.log"
        samples_logger = logging.getLogger('samples')
        samples_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        samples_logger.handlers.clear()
        
        file_handler = logging.FileHandler(samples_log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        samples_logger.addHandler(file_handler)
        
        # Prediction artifact for wandb
        if trainer.use_wandb:
            prediction_artifact = wandb.Artifact(
                name=f"generated_samples", 
                type="predictions", 
                description="Model generated samples"
            )
        
        with torch.no_grad():
            generated_samples = []
            for i, batch in enumerate(trainer.valid_loader):
                if i >= num_samples // samples_per_batch:
                    break
                
                source_ids = batch['source_text'].to(trainer.device)
                target_ids = batch['target_text'].to(trainer.device)
                
                # Generate prediction without teacher forcing
                outputs = trainer.model(src=source_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1).cpu().numpy()
                
                for idx in range(min(len(predictions), samples_per_batch)):
                    src_tokens = [t for t in source_ids[idx].cpu().numpy() if t != 0]
                    tgt_tokens = [t for t in target_ids[idx].cpu().numpy() if t != 0]
                    pred_tokens = [t for t in predictions[idx] if t != 0]
                    
                    # Decode tokens to text
                    src_text = trainer.train_dataset.tokenizer.decode(src_tokens)
                    tgt_text = trainer.train_dataset.tokenizer.decode(tgt_tokens)
                    pred_text = trainer.train_dataset.tokenizer.decode(pred_tokens)
                    
                    # Calculate BLEU score
                    sample_bleu = trainer.metrics.compute_bleu_score([pred_tokens], [tgt_tokens])
                    
                    # Log sample details
                    log_message = (
                        f"Sample {i * samples_per_batch + idx + 1}: "
                        f"Source: {src_text}, "
                        f"Target: {tgt_text}, "
                        f"Prediction: {pred_text}, "
                        f"BLEU: {sample_bleu:.4f}"
                    )
                    samples_logger.info(log_message)
                    
                    # Store for potential further use
                    generated_samples.append({
                        'source': src_text,
                        'target': tgt_text,
                        'prediction': pred_text,
                        'bleu': sample_bleu
                    })
        
        # Log artifact to wandb if using wandb
        if trainer.use_wandb and generated_samples:
            # Convert samples to DataFrame for easier logging
            import pandas as pd
            samples_df = pd.DataFrame(generated_samples)
            samples_df.to_csv(samples_log_path.with_suffix('.csv'), index=False)
            prediction_artifact.add_file(samples_log_path.with_suffix('.csv'))
            wandb.log_artifact(prediction_artifact)
        
        return generated_samples
