# src/trainers/autoencoder_trainer.py
from .base_trainer import BaseTrainer
from ..models.text2text_autoencoders import PositionalAutoencoder

class AutoencoderTrainer(BaseTrainer):
    def initialize_model(self):
        return PositionalAutoencoder(**self.model_config)
