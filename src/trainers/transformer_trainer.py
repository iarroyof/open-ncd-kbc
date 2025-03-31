# src/trainers/transformer_trainer.py
from .base_trainer import BaseTrainer
from ..models.text2text_autoencoders import VanillaTransformer

class TransformerTrainer(BaseTrainer):
    def initialize_model(self):
        return VanillaTransformer(**self.model_config)
