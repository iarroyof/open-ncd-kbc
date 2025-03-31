# src/trainers/attention_lstm_trainer.py
from .base_trainer import BaseTrainer
from ..models.text2text_autoencoders import AttentionLSTMSeq2Seq

class AttentionLSTMTrainer(BaseTrainer):
    def initialize_model(self):
        return AttentionLSTMSeq2Seq(**self.model_config)
