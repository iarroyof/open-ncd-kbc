# src/trainers/attention_gru_trainer.py
from .base_trainer import BaseTrainer
from ..models.text2text_autoencoders import AttentionGRUModel

class AttentionGRUTrainer(BaseTrainer):
    def initialize_model(self):
        return AttentionGRUModel(**self.model_config)

