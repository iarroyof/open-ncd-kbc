# prediction_logging.py
import logging
import torch
import os

class PredictionLogger:
    @staticmethod
    def setup_prediction_logger(log_dir, filename='predictions.log'):
        """
        Create a dedicated logger for predictions
        
        Args:
            log_dir (Path): Directory to store log files
            filename (str, optional): Name of the log file
        
        Returns:
            logging.Logger: Configured logger for predictions
        """
        log_path = log_dir / filename
        
        # Create logger
        prediction_logger = logging.getLogger('predictions')
        prediction_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to prevent duplicate logging
        prediction_logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        prediction_logger.addHandler(file_handler)
        
        return prediction_logger

    @staticmethod
    def log_batch_predictions(
        prediction_logger, 
        tokenizer, 
        metrics_calculator,
        batch, 
        outputs, 
        batch_idx, 
        phase="train", 
        num_samples=3
    ):
        """
        Log predictions for a batch of samples
        
        Args:
            prediction_logger (logging.Logger): Logger for predictions
            tokenizer: Tokenizer to decode tokens
            metrics_calculator: Metrics calculator for BLEU score
            batch (dict): Input batch with source and target texts
            outputs (torch.Tensor): Model predictions
            batch_idx (int): Current batch index
            phase (str, optional): Training phase
            num_samples (int, optional): Number of samples to log
        """
        try:
            # Convert outputs to token indices
            predictions = outputs.argmax(dim=-1).cpu().numpy()
            source_ids = batch['source_text'].cpu().numpy()
            target_ids = batch['target_text'].cpu().numpy()
            
            # Sample up to num_samples examples from the batch
            batch_size = len(predictions)
            num_samples = min(num_samples, batch_size)
            
            for idx in range(num_samples):
                # Decode non-padding tokens
                src_tokens = [t for t in source_ids[idx] if t != 0]
                tgt_tokens = [t for t in target_ids[idx] if t != 0]
                pred_tokens = [t for t in predictions[idx] if t != 0]
                
                # Decode tokens to text
                src_text = tokenizer.decode(src_tokens)
                tgt_text = tokenizer.decode(tgt_tokens)
                pred_text = tokenizer.decode(pred_tokens)
                
                # Calculate BLEU score for this sample
                sample_bleu = metrics_calculator.compute_bleu_score([pred_tokens], [tgt_tokens])
                
                # Log prediction details
                log_message = (
                    f"Phase: {phase}, "
                    f"Batch: {batch_idx}, "
                    f"Source: {src_text}, "
                    f"Target: {tgt_text}, "
                    f"Prediction: {pred_text}, "
                    f"BLEU: {sample_bleu:.4f}"
                )
                prediction_logger.info(log_message)
        except Exception as e:
            logging.warning(f"Error logging predictions: {str(e)}")

    @staticmethod
    def generate_samples(
        model, 
        valid_loader, 
        tokenizer, 
        metrics_calculator, 
        log_dir, 
        device, 
        num_samples=10,
        filename='generated_samples.log'
    ):
        """
        Generate and log prediction samples from the validation dataset
        
        Args:
            model (nn.Module): Trained model
            valid_loader (DataLoader): Validation data loader
            tokenizer: Tokenizer to decode tokens
            metrics_calculator: Metrics calculator for BLEU score
            log_dir (Path): Directory to store log files
            device (torch.device): Device to run inference on
            num_samples (int, optional): Number of samples to generate
            filename (str, optional): Name of the log file
        """
        model.eval()
        samples_log_path = log_dir / filename
        samples_logger = logging.getLogger('samples')
        samples_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        samples_logger.handlers.clear()
        
        file_handler = logging.FileHandler(samples_log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        samples_logger.addHandler(file_handler)
        
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                if i >= num_samples:
                    break
                
                source_ids = batch['source_text'].to(device)
                target_ids = batch['target_text'].to(device)
                
                # Generate prediction without teacher forcing
                outputs = model(src=source_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1).cpu().numpy()
                
                for idx in range(min(len(predictions), 3)):  # Up to 3 samples per batch
                    src_tokens = [t for t in source_ids[idx].cpu().numpy() if t != 0]
                    tgt_tokens = [t for t in target_ids[idx].cpu().numpy() if t != 0]
                    pred_tokens = [t for t in predictions[idx] if t != 0]
                    
                    # Decode tokens to text
                    src_text = tokenizer.decode(src_tokens)
                    tgt_text = tokenizer.decode(tgt_tokens)
                    pred_text = tokenizer.decode(pred_tokens)
                    
                    # Calculate BLEU score
                    sample_bleu = metrics_calculator.compute_bleu_score([pred_tokens], [tgt_tokens])
                    
                    # Log sample details
                    log_message = (
                        f"Sample {i * 3 + idx + 1}: "
                        f"Source: {src_text}, "
                        f"Target: {tgt_text}, "
                        f"Prediction: {pred_text}, "
                        f"BLEU: {sample_bleu:.4f}"
                    )
                    samples_logger.info(log_message)
