import torch.nn as nn
from typing import Dict

from .text2text_autoencoders import (
    VanillaTransformer,
    ConvS2S,
    AttentionLSTMSeq2Seq,
    AttentionGRUModel,
    PositionalAutoencoder,
)

SUPPORTED_MODELS = {
    'transformer': VanillaTransformer,
    'conv_s2s': ConvS2S,
    'attention_lstm': AttentionLSTMSeq2Seq,
    'attention_gru': AttentionGRUModel,
    'autoencoder': PositionalAutoencoder,
}

def build_model(model_type: str, config: Dict) -> nn.Module:
    """
    Instantiates and returns the appropriate model class.

    Args:
        model_type (str): Key representing the model (e.g., "transformer", "conv_s2s").
        config (Dict): Model configuration dictionary.

    Returns:
        nn.Module: Instantiated model object.
    """
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Available options: {list(SUPPORTED_MODELS.keys())}")

    model_class = SUPPORTED_MODELS[model_type]
    return model_class(**config)
