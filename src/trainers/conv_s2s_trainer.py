# src/trainers/conv_s2s_trainer.py
from .base_trainer import BaseTrainer
from ..models.text2text_autoencoders import ConvS2S

class ConvS2STrainer(BaseTrainer):
    def initialize_model(self):
        return ConvS2S(**self.model_config)
