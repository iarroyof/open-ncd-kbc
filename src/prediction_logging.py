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
        if not hasattr(trainer, 'prediction_logger') or \
           not hasattr(trainer, 'train_dataset') or \
           not hasattr(trainer.train_dataset, 'tokenizer'):
            return
        
        try:
            predictions = outputs.argmax(dim=-1).cpu()
            source_ids = batch['source_text'].cpu()
            target_ids = batch['target_text'].cpu()
            eos_token_id = trainer.train_dataset.tokenizer.token_to_id("[EOS]")
            batch_size = len(predictions)
            num_samples = min(max_samples, batch_size)
            sample_indices = random.sample(range(batch_size), num_samples)
            current_epoch = trainer.current_epoch
            
            for idx in sample_indices:
                src_seq = source_ids[idx]
                tgt_seq = target_ids[idx]
                pred_seq = predictions[idx]
                
                src_trim = trainer.trim_sequence_at_eos(src_seq, eos_token_id)
                tgt_trim = trainer.trim_sequence_at_eos(tgt_seq, eos_token_id)
                pred_trim = trainer.trim_sequence_at_eos(pred_seq, eos_token_id)
                
                # Decode tokens to text (assumes tokenizer.decode can take a list of ints)
                src_text = trainer.train_dataset.tokenizer.decode(src_trim)
                tgt_text = trainer.train_dataset.tokenizer.decode(tgt_trim)
                pred_text = trainer.train_dataset.tokenizer.decode(pred_trim)
                
                # Compute metrics for the trimmed sequences
                metrics_result = trainer.metrics.compute_metrics(
                    [torch.tensor(pred_trim)],
                    [torch.tensor(tgt_trim)]
                )
                sample_bleu = metrics_result.get("bleu", 0.0)
                
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
                
                if trainer.use_wandb:
                    wandb.log({
                        f"{phase}_predictions": wandb.Table(
                            columns=["Epoch", "Phase", "Batch", "Source", "Target", "Prediction", "BLEU"],
                            data=[[current_epoch, phase, batch_idx, src_text, tgt_text, pred_text, sample_bleu]]
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
        trainer.model.eval()
        samples_log_path = trainer.log_dir / "generated_samples.log"
        samples_logger = logging.getLogger('samples')
        samples_logger.setLevel(logging.INFO)
        samples_logger.handlers.clear()
        file_handler = logging.FileHandler(samples_log_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        samples_logger.addHandler(file_handler)
        
        if trainer.use_wandb:
            prediction_artifact = wandb.Artifact(
                name=f"generated_samples", 
                type="predictions", 
                description="Model generated samples"
            )
        
        eos_token_id = trainer.train_dataset.tokenizer.token_to_id("[EOS]")
        generated_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(trainer.valid_loader):
                if i >= num_samples // samples_per_batch:
                    break
                
                source_ids = batch['source_text'].to(trainer.device)
                target_ids = batch['target_text'].to(trainer.device)
                
                outputs = trainer.model(src=source_ids, teacher_forcing_ratio=0.0)
                predictions = outputs.argmax(dim=-1).cpu()
                
                for idx in range(min(len(predictions), samples_per_batch)):
                    src_seq = source_ids[idx].cpu()
                    tgt_seq = target_ids[idx].cpu()
                    pred_seq = predictions[idx]
                    
                    src_trim = trainer.trim_sequence_at_eos(src_seq, eos_token_id)
                    tgt_trim = trainer.trim_sequence_at_eos(tgt_seq, eos_token_id)
                    pred_trim = trainer.trim_sequence_at_eos(pred_seq, eos_token_id)
                    
                    src_text = trainer.train_dataset.tokenizer.decode(src_trim)
                    tgt_text = trainer.train_dataset.tokenizer.decode(tgt_trim)
                    pred_text = trainer.train_dataset.tokenizer.decode(pred_trim)
                    
                    metrics_result = trainer.metrics.compute_metrics(
                        [torch.tensor(pred_trim)],
                        [torch.tensor(tgt_trim)]
                    )
                    sample_bleu = metrics_result.get("bleu", 0.0)
                    
                    log_message = (
                        f"Sample {i * samples_per_batch + idx + 1}: "
                        f"Source: {src_text}, "
                        f"Target: {tgt_text}, "
                        f"Prediction: {pred_text}, "
                        f"BLEU: {sample_bleu:.4f}"
                    )
                    samples_logger.info(log_message)
                    generated_samples.append({
                        'source': src_text,
                        'target': tgt_text,
                        'prediction': pred_text,
                        'bleu': sample_bleu
                    })
        
        if trainer.use_wandb and wandb.run is not None and generated_samples:
            import pandas as pd
            samples_df = pd.DataFrame(generated_samples)
            csv_path = samples_log_path.with_suffix('.csv')
            samples_df.to_csv(csv_path, index=False)
            prediction_artifact.add_file(str(csv_path))
            wandb.log_artifact(prediction_artifact)
        
        return generated_samples

    @staticmethod
    def log_evaluation_samples(trainer, sample_predictions):
        """
        Log a set of evaluation sample predictions that were collected during evaluation.
        This function is called once per evaluation run.
        
        Args:
            trainer: The trainer instance.
            sample_predictions (list of dict): Each dict should have 'source', 'target', and 'prediction' keys.
        """
        if not sample_predictions:
            return
    
        # Log locally using the prediction logger
        for i, sample in enumerate(sample_predictions):
            log_message = (
                f"Evaluation Sample {i+1}: "
                f"Source: {sample['source']}, "
                f"Target: {sample['target']}, "
                f"Prediction: {sample['prediction']}"
            )
            trainer.prediction_logger.info(log_message)
        
        # Log artifact to wandb only if a run is active
        if trainer.use_wandb and wandb.run is not None:
            import pandas as pd
            samples_log_path = trainer.log_dir / "evaluation_samples.log"
            # Save samples in CSV format
            df = pd.DataFrame(sample_predictions)
            csv_path = samples_log_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            # Create an artifact and log it
            eval_artifact = wandb.Artifact(
                name="evaluation_samples", 
                type="predictions", 
                description="Evaluation sample predictions"
            )
            eval_artifact.add_file(str(csv_path))
            wandb.log_artifact(eval_artifact)
