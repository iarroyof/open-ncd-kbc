# Standard library imports
import os
import time
import random
import argparse
import logging
import functools
import re
import string
import math

# Numerical and data processing libraries
import numpy as np
import pandas as pd

# Typing utilities
import typing
from typing import Any, Tuple, Dict, Optional, List

# TensorFlow and Keras imports
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"       # NVIDIA perf tip
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# ── experiment tracking ───────────────────────────────────────────────────
import wandb

# In TF 2.10+, preprocessing is no longer in experimental
from tensorflow.keras.layers import TextVectorization

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

# Global constants
CS_LABELS = False
STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")


# --- Utility Functions ---

# ── wandb_helper.py ──────────────────────────────────────────────────────────

def make_overfit_callback(total_epochs: int,
                          a: float = 12.,
                          b: float = 8.,
                          c: float = -4.):
    """
    Returns a Keras Callback that, every epoch,
      • computes rel_gap, epoch_ratio, p_overfit
      • reads AUROC from logs (added via model.compile)
      • logs everything to Weights & Biases.

    Parameters
    ----------
    total_epochs : int
        The number of training epochs you passed to `model.fit(...)`
        so epoch_ratio = (epoch + 1) / total_epochs is in [0, 1].
    a, b, c : float
        Coefficients of the logistic formula.
        p_overfit = sigmoid(a*rel_gap + b*epoch_ratio + c).
    """
    class OverfitLogger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_loss = logs.get("loss")
            val_loss   = logs.get("val_loss")
            if train_loss is None or val_loss is None:
                return                                  # can’t compute

            rel_gap     = (val_loss - train_loss) / max(train_loss, 1e-8)
            epoch_ratio = (epoch + 1) / total_epochs
            z           = a * rel_gap + b * epoch_ratio + c
            p_overfit   = 1. / (1. + math.exp(-z))

            # fetch AUROC if the metric is present
            auroc = logs.get("val_auroc")              # name in `compile(...)`
            wandb.log({
                "epoch":        epoch + 1,
                "rel_gap":      rel_gap,
                "epoch_ratio":  epoch_ratio,
                "p_overfit":    p_overfit,
                "val_auroc":    auroc
            }, step=epoch)                             # 1 log per epoch

    return OverfitLogger()

class ShapeChecker:
    """Utility class to check tensor shapes during execution."""
    def __init__(self):
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return
        if isinstance(names, str):
            names = (names,)
        shape = tf.shape(tensor)
        rank = tf.rank(tensor)
        if rank != len(names):
            raise ValueError(f'Rank mismatch: found {rank}: {shape.numpy()} expected {len(names)}: {names}')
        for i, name in enumerate(names):
            old_dim = name if isinstance(name, int) else self.shapes.get(name, None)
            new_dim = shape[i]
            if broadcast and new_dim == 1:
                continue
            if old_dim is None:
                self.shapes[name] = new_dim
                continue
            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}' found: {new_dim} expected: {old_dim}")


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    """Standardize input strings by converting to lowercase and removing punctuation."""
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(STRIP_CHARS)}]", "")


def prepare_data(line, start_token='[start] ', end_token=' [end]', pmid=True,
                 include_labels=False, include_sent=False, all_start_end=False):
    """
    Process a TSV line into input and output phrases for training.
    (Identical to original script)
    """
    line = line.split('\t')
    if pmid:
        line.pop(0)  # Remove PMID column
    pred = ' '.join(re.findall('[A-Z][a-z]*', line[1])).lower() or line[1]
    if not line[4].strip().isdigit() and not re.match(r'^-?\d+(?:\.\d+)$', line[4].strip()):
        complements = []
        i = 4
        while not line[i].isdigit(): # Check if index i is valid
            if i >= len(line): # break if i is out of bounds
                break
            complements.append(line[i])
            line.pop(i)
        line[3] = " ".join([line[3]] + complements)
    sample = [line[0], pred, line[2], f"{start_token}{line[3]}{end_token}", float(line[4].strip())]
    if not include_labels:
        del sample[-1]
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])
    if not include_sent:
        del sample[0]
        sample_i = ' '.join([sample[1], sample[0]])
        if all_start_end:
            sample_i = f"{start_token}{sample_i}{end_token}"
    else:
        sample_i = ' '.join([sample[0], sample[2], sample[1]])
    return sample_i, sample_o


def sort_cols(columns):
    """Sort columns based on their last two characters."""
    ends = np.unique([c[-2:] for c in columns])
    return [c for e in ends for c in columns if c.endswith(e)]


def load_vectorizer(from_file):
    """
    Load a saved TextVectorization layer from disk.
    (Identical to original script)
    """
    model = tf.keras.models.load_model(from_file, custom_objects={"custom_standardization": custom_standardization})
    vocab = model.layers[0].get_vocabulary()
    config = model.layers[0].get_config()
    vectorizer = TextVectorization.from_config(config)
    vectorizer.adapt(['Initializing vectorizer'])
    vectorizer.set_vocabulary(vocab)
    return vectorizer


def save_vectorizer(vectorizer, to_file):
    """Save a TextVectorization layer to disk."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer
    ])
    model.compile() # Compile before saving a Sequential model containing only a layer
    model.save(to_file)


def parse_dataset_name(training_data):
    """Parse dataset name from the training data file path."""
    training_data = training_data.split(os.sep)[-1].lower()
    names_dic = {"NCD": 'ncd' in training_data, "GP": 'gp' in training_data, "CN": 'conceptnet' in training_data}
    return '-'.join(k for k, v in names_dic.items() if v)

# --- Transformer Model Architecture ---

def create_padding_mask(seq):
    """Creates a boolean padding mask for sequences.
    Args:
        seq: A tensor of shape (batch_size, seq_len) containing token IDs.
             0 is assumed to be the padding token.
    Returns:
        A boolean tensor of shape (batch_size, 1, 1, seq_len) where True means masked.
    """
    seq_mask = tf.math.equal(seq, 0)  # This is already boolean: True where padded
    return seq_mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """Creates a boolean look-ahead mask for the decoder's self-attention.
    Args:
        size: The length of the target sequence.
    Returns:
        A boolean tensor of shape (size, size) where True means future tokens are masked.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  # float: 1.0 for upper triangle (masked)
    return tf.cast(mask, tf.bool)  # Convert to boolean: True for upper triangle
    
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name='masked_loss')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def __call__(self, real, pred):
        pred = tf.cast(pred, tf.float32) # Ensure float32 for stability
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(tf.math.not_equal(real, 0), loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask) # Average over non-padded tokens
        
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(position, d_model)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs): # inputs: (batch_size, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]
        # Ensure positional encoding matches the dtype of inputs for mixed precision
        return inputs + tf.cast(self.pos_encoding[:, :seq_len, :], dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config

class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()


    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # Self-attention. The MultiHeadAttention layer should handle its inputs correctly in a @tf.function context.
        # Pass the boolean mask directly. The LayerNormalization also expects the float dtype determined by mixed precision.
        # The explicit casts were removed as they are often handled by mixed precision.

        attn, _ = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training, return_attention_scores=False)

        # Residual connection and normalization
        out1 = self.norm1(x + attn)

        ffn_output = self.ffn(out1, training=training)

        # Residual connection and normalization
        out2 = self.norm2(out1 + ffn_output)
        return out2


class DecBlock(layers.Layer):
    def __init__(
        self,
        dim: int,
        latent: int,
        heads: int,
        key_dim: int,
    ):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(heads, key_dim) # Causal self-attention
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim) # Encoder-decoder attention

        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

    def call(
        self,
        y: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool = False,
        look_ahead_mask: tf.Tensor | None = None, # Added mask args to call signature
        padding_mask: tf.Tensor | None = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: # Also return attention weights
        # Unmasked self-attention (apply look_ahead_mask)
        # Removed explicit casts here, rely on mixed precision if enabled
        attn1, attn_weights_block1 = self.self_mha(query=y, value=y, key=y, attention_mask=look_ahead_mask, training=training, return_attention_scores=True)
        out1 = self.norm1(y + attn1)

        # Unmasked cross-attention (apply padding_mask to encoder output)
        # Removed explicit casts here, rely on mixed precision if enabled
        attn2, attn_weights_block2 = self.cross_mha(query=out1, value=enc_output, key=enc_output, attention_mask=padding_mask, training=training, return_attention_scores=True)
        out2 = self.norm2(out1 + attn2)

        ffn_output = self.ffn(out2, training=training)
        out3 = self.norm3(out2 + ffn_output)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        # Corrected: Use the previously defined EncoderLayer, not EncBlock
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) # x is now potentially float16 if mixed precision is on
        # Scale embedding by d_model
        scaling_factor = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x *= tf.cast(scaling_factor, x.dtype) # Cast scaling factor to x's dtype

        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            # Pass the mask to each EncoderLayer
            x = self.enc_layers[i](x, training, mask)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers, "d_model": self.d_model,
            "num_heads": self.num_heads, "dff": self.dff,
            "input_vocab_size": self.input_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "rate": self.rate
        })
        return config

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        # Corrected: Use the previously defined DecoderLayer, not DecBlock
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        # Ensure the final layer computes in float32 for numerical stability with mixed precision
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, dtype=tf.float32)


    # Added mask arguments to call signature
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) # x is now potentially float16
        # Scale embedding by d_model
        scaling_factor = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x *= tf.cast(scaling_factor, x.dtype) # Cast scaling factor to x's dtype

        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            # Pass both masks to each DecoderLayer
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1_self_attn'] = block1
            attention_weights[f'decoder_layer{i+1}_block2_enc_dec_attn'] = block2

        logits = self.final_layer(x) # Output logits are float32 due to dtype=tf.float32

        class TransformerDecoderOutput: # Simple container
            def __init__(self, logits_val, attention_weights_val):
                self.logits = logits_val
                self.attention_weights = attention_weights_val

        return TransformerDecoderOutput(logits, attention_weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers, "d_model": self.d_model,
            "num_heads": self.num_heads, "dff": self.dff,
            "target_vocab_size": self.target_vocab_size,
            "maximum_position_encoding": self.maximum_position_encoding,
            "rate": self.rate
        })
        return config

class TrainTranslator(tf.keras.Model):
    def __init__(self, d_model: int, num_heads: int, dff: int, num_layers: int,
                 input_text_processor, output_text_processor,
                 maximum_position_encoding: int, dropout: float = 0.1):
        super().__init__()

        self.input_vocab_size = input_text_processor.vocabulary_size()
        self.output_vocab_size = output_text_processor.vocabulary_size()

        # Pass correct layer classes
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               self.input_vocab_size, maximum_position_encoding, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               self.output_vocab_size, maximum_position_encoding, dropout)


        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.train_accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.val_accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name="val_accuracy")

    def _create_masks(self, inp, tar_inp_seq):
        # Encoder padding mask (boolean: True for pads)
        enc_padding_mask = create_padding_mask(inp)

        # Decoder's 2nd MHA: masks padding in the encoder's output (keys/values)
        dec_enc_padding_mask = create_padding_mask(inp)

        # Decoder's 1st MHA (self-attention):
        target_seq_len = tf.shape(tar_inp_seq)[1]

        # Look-ahead mask (boolean: True for future positions)
        look_ahead_mask_for_shape = create_look_ahead_mask(target_seq_len)

        # Padding mask for the target sequence itself (boolean: True for pads in target)
        dec_target_padding_mask = create_padding_mask(tar_inp_seq)

        # Combined mask for decoder self-attention.
        combined_dec_self_attention_mask = tf.logical_or(
            dec_target_padding_mask,
            look_ahead_mask_for_shape[tf.newaxis, tf.newaxis, :, :]
        )
        return enc_padding_mask, combined_dec_self_attention_mask, dec_enc_padding_mask

    @tf.function # Moved the core logic into a tf.function
    def _call_logic(self, input_tokens, target_tokens_input, training):
        enc_padding_mask, combined_dec_self_attn_mask, dec_enc_padding_mask = self._create_masks(
            input_tokens, target_tokens_input
        )
        # Pass masks to encoder and decoder layers within the @tf.function
        enc_output = self.encoder(input_tokens, training=training, mask=enc_padding_mask)
        decoder_output_obj = self.decoder(
            target_tokens_input, enc_output, training=training,
            look_ahead_mask=combined_dec_self_attn_mask,
            padding_mask=dec_enc_padding_mask
        )
        return decoder_output_obj.logits

    def train_step(self, data):
        input_text, target_text = data
        # Text processing happens outside the @tf.function decorated _call_logic
        # because TextVectorization can be slow inside tf.function.
        # Keras's .fit handles calling these preprocessing steps outside the main training loop trace.
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        tar_inp = target_tokens[:, :-1]
        tar_real = target_tokens[:, 1:]

        with tf.GradientTape() as tape:
            # Call the @tf.function wrapped core logic
            predictions = self._call_logic(input_tokens, tar_inp, training=True)
            loss = self.compiled_loss(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_accuracy_metric.update_state(tar_real, predictions, sample_weight=tf.cast(tar_real != 0, tf.float32))
        # Return a dict mapping metric names to current value
        results = {'loss': loss, 'accuracy': self.train_accuracy_metric.result()}
        # Include other metrics from compile()
        for metric in self.metrics:
            # Check if the metric is one of the compiled metrics and not the ones we handle manually
            if metric.name in [m.name for m in self.compiled_metrics.metrics]:
                 results[metric.name] = metric.result()
        return results


    def test_step(self, data):
        input_text, target_text = data
        # Text processing happens outside the @tf.function wrapped _call_logic
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        tar_inp = target_tokens[:, :-1]
        tar_real = target_tokens[:, 1:]

        # Call the @tf.function wrapped core logic
        predictions = self._call_logic(input_tokens, tar_inp, training=False)
        loss = self.compiled_loss(tar_real, predictions)

        self.val_accuracy_metric.update_state(tar_real, predictions, sample_weight=tf.cast(tar_real != 0, tf.float32))
        results = {'loss': loss, 'accuracy': self.val_accuracy_metric.result()}
        for metric in self.metrics: # Include other metrics from compile()
            if metric.name in [m.name for m in self.compiled_metrics.metrics]:
                 results[metric.name] = metric.result()
        return results

    @property
    def metrics(self):
        # We need to list all metrics here for Keras to manage them,
        # including the ones added in `compile()`.
        # `compiled_loss` and `compiled_metrics` are set by `compile`.
        # The accuracy metrics are manually managed here for train/val.
        # Ensure unique metrics.
        base_metrics = [self.train_accuracy_metric, self.val_accuracy_metric]
        if self._is_compiled: # if compiled
             # Add compiled metrics, avoiding duplicates if they exist in base_metrics by chance
             compiled_metrics_names = {m.name for m in self.compiled_metrics.metrics}
             base_metrics.extend([m for m in self.compiled_metrics.metrics if m.name not in {bm.name for bm in base_metrics}])
        return base_metrics


    def get_config(self):
        # get_config for the model should return config for its layers if needed for saving/loading.
        # For simplicity and since we save vectorizers separately and the model's weights,
        # we can return a basic config or rely on the layers' configs if they are custom and serializable.
        # The current simple config is likely sufficient if we reload using the same parameters.
        return {
            "d_model": self.encoder.d_model,
            "num_heads": self.encoder.num_heads,
            "dff": self.encoder.dff,
            "num_layers": self.encoder.num_layers,
            "dropout_rate": self.encoder.rate,
            "maximum_position_encoding": self.encoder.maximum_position_encoding
            # Note: input/output_text_processor are not directly part of the model's saveable config
            # when using save_weights_only. They are handled separately.
        }
        
class Translator(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor, output_text_processor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        
        self.output_token_string_from_index = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='', invert=True)
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        
        token_mask_ids = index_from_string(['', '[UNK]', '[start]']).numpy() # Don't predict [start] after first step
        self.token_mask = np.zeros(index_from_string.vocabulary_size(), dtype=bool)
        self.token_mask[token_mask_ids] = True
        self.start_token = index_from_string(tf.constant('[start]'))
        self.end_token = index_from_string(tf.constant('[end]'))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        result_text = tf.strings.regex_replace(result_text, '^ *(\\[start\\])+ *', '')
        result_text = tf.strings.regex_replace(result_text, ' *(\\[end\\])+ *$', '')
        result_text = tf.strings.strip(result_text)
        return result_text

    def sample(self, logits, temperature):
        logits = tf.cast(logits, tf.float32)
        mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(mask, tf.constant(-np.inf, dtype=tf.float32), logits)
        if temperature == 0.0:
            return tf.argmax(logits, axis=-1, output_type=tf.int64)
        logits = tf.squeeze(logits, axis=1)
        return tf.random.categorical(logits / temperature, num_samples=1, dtype=tf.int64)

    def translate(self, input_text, *, max_length=50, return_attention=True, temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_padding_mask = create_padding_mask(input_tokens)
        enc_output = self.encoder(input_tokens, training=False, mask=enc_padding_mask)

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, tf.cast(tf.fill([batch_size], self.start_token), dtype=tf.int64))
        
        attention_step_history = [] # To store attention from each decoding step for one head

        for i in tf.range(max_length):
            output_so_far = tf.transpose(output_array.stack()) # (batch, current_dec_seq_len)
            
            look_ahead_mask_for_step = create_look_ahead_mask(tf.shape(output_so_far)[1])
            # dec_padding_mask for MHA2 (masking encoder output) is enc_padding_mask
            
            dec_output_obj = self.decoder(output_so_far, enc_output, training=False,
                                          look_ahead_mask=look_ahead_mask_for_step,
                                          padding_mask=enc_padding_mask)
            
            predictions_for_last_token = dec_output_obj.logits[:, -1:, :] # (batch, 1, vocab_size)
            
            if return_attention:
                # Store enc-dec attention for the *last predicted token* over all encoder tokens.
                # Attention dict: key like f'decoder_layer{N}_block2_enc_dec_attn'
                # Shape: (batch, num_heads, current_dec_seq_len, enc_seq_len)
                last_layer_key = f'decoder_layer{self.decoder.num_layers}_block2_enc_dec_attn'
                current_step_attention = dec_output_obj.attention_weights[last_layer_key] # (B, H, T_dec, T_enc)
                # Attention for the last decoder token (the one whose logits we are using)
                attention_for_new_token = current_step_attention[:, :, -1, :] # (B, H, T_enc)
                # Average over heads for simplicity in AttentionLogger
                avg_head_attention = tf.reduce_mean(attention_for_new_token, axis=1) # (B, T_enc)
                attention_step_history.append(avg_head_attention[:, tf.newaxis, :]) # (B, 1, T_enc)

            predicted_id = self.sample(predictions_for_last_token, temperature) # (batch, 1)
            predicted_id = tf.squeeze(predicted_id, axis=1) # (batch,)
            output_array = output_array.write(i + 1, predicted_id)

            if tf.reduce_all(tf.math.logical_or(predicted_id == self.end_token, predicted_id == 0)): # Also stop on padding
                break
        
        result_tokens = tf.transpose(output_array.stack())
        result_tokens = result_tokens[:, 1:] # Exclude the initial start_token

        result_text = self.tokens_to_text(result_tokens)
        
        output_dict = {'text': result_text}
        if return_attention:
            if attention_step_history:
                final_attention_tensor = tf.concat(attention_step_history, axis=1) # (B, final_dec_T, T_enc)
            else:
                enc_seq_len = tf.shape(input_tokens)[1]
                final_attention_tensor = tf.zeros((batch_size, 0, enc_seq_len), dtype=tf.float32)
            output_dict['attention'] = final_attention_tensor
            output_dict['result_tokens'] = result_tokens # For AttentionLogger
            
        return output_dict

    @tf.function(input_signature=[
        tf.TensorSpec(dtype=tf.string, shape=[None]), # input_text
        tf.TensorSpec(dtype=tf.int32, shape=[]),      # max_length
        tf.TensorSpec(dtype=tf.bool, shape=[]),       # return_attention
        tf.TensorSpec(dtype=tf.float32, shape=[])     # temperature
    ])
    def tf_translate(self, input_text, max_length, return_attention, temperature):
        return self.translate(input_text, max_length=max_length, 
                               return_attention=return_attention, temperature=temperature)

class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.logs = []
    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])

class AttentionLogger(keras.callbacks.Callback):
    def __init__(self, translator, sample_sentence: str):
        super().__init__()
        self.translator = translator
        self.sample_sentence = tf.constant([sample_sentence])
        self.src_lookup = tf.keras.layers.StringLookup(
            vocabulary=translator.input_text_processor.get_vocabulary(),
            mask_token='', invert=True)
        self.tgt_lookup = tf.keras.layers.StringLookup(
            vocabulary=translator.output_text_processor.get_vocabulary(),
            mask_token='', invert=True)

    def on_epoch_end(self, epoch, logs=None):
        out = self.translator.translate(self.sample_sentence, max_length=50, return_attention=True, temperature=0.0) # Greedy
        
        attn_tensor = out.get('attention')
        result_tokens_tensor = out.get('result_tokens')

        if attn_tensor is None or result_tokens_tensor is None:
            logging.warning("AttentionLogger: Attention or result_tokens not found in translator output. Skipping plot.")
            return

        attn = tf.squeeze(attn_tensor, 0) # (dec_T, enc_T)
        
        if tf.rank(attn) != 2 or attn.shape[0] == 0: # Allow enc_T to be 0 if input is empty for some reason
            # If dec_T is 0, means no output was generated beyond start (which is stripped).
            # If enc_T is 0 (from attn.shape[1]), then source was empty.
            logging.warning(f"AttentionLogger: Attention matrix for plot is empty or has unexpected rank. Shape: {attn.shape}. Skipping plot.")
            return

        src_ids = self.translator.input_text_processor(self.sample_sentence)[0].numpy()
        tgt_ids_np = result_tokens_tensor[0].numpy()

        enc_len = int((src_ids != 0).sum())
        dec_len = int((tgt_ids_np != 0).sum()) 

        if dec_len == 0 or enc_len == 0 : # If either is zero, plot is meaningless
            logging.info(f"AttentionLogger: enc_len ({enc_len}) or dec_len ({dec_len}) is zero. Skipping plot.")
            return

        final_attn = attn.numpy()[:dec_len, :enc_len] # Slice attention to actual lengths
        final_src_ids = src_ids[:enc_len]
        final_tgt_ids = tgt_ids_np[:dec_len]
        
        if final_attn.shape[0] == 0 or final_attn.shape[1] == 0:
             logging.warning(f"AttentionLogger: Attention matrix became empty after trimming. dec_len={dec_len}, enc_len={enc_len}. Skipping plot.")
             return

        src_words = self.src_lookup(final_src_ids).numpy().astype(str)
        tgt_words = self.tgt_lookup(final_tgt_ids).numpy().astype(str)
        
        fig, ax = plt.subplots(figsize=(max(6, enc_len * .4 + 1.5), max(6, dec_len * .4 + 1.5))) # Adjusted fig size
        im = ax.imshow(final_attn, aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(enc_len)) # Use np.arange for explicit tick positions
        ax.set_xticklabels(src_words, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(dec_len))
        ax.set_yticklabels(tgt_words, fontsize=8)
        ax.set_xlabel("Encoder tokens")
        ax.set_ylabel("Decoder tokens")
        fig.colorbar(im, ax=ax, fraction=.046, pad=0.04)
        fig.tight_layout()
        
        step = self.model.optimizer.iterations.numpy()
        wandb.log({"attention_matrix": wandb.Image(fig)}, step=int(step))
        plt.close(fig)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Train a Transformer sequence-to-sequence model.")
    parser.add_argument("-s", "--seqLen", type=int, default=50, help="Per-sample sequence length")
    # nSteps is now d_model for Transformer
    parser.add_argument("-u", "--nSteps", type=int, default=256, help="Model dimensionality (d_model)") # Adjusted default for typical Transformer
    parser.add_argument("-f", "--nFeatures", type=int, default=15000, help="Maximum vocabulary size")
    parser.add_argument("-b", "--batchSize", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--nEpochs", type=int, default=40, help="Number of training epochs")
    # embeddingDim is d_model for Transformer. Should be same as nSteps.
    parser.add_argument("-d", "--embeddingDim", type=int, default=256, help="Word embedding dimensionality (should match d_model)")
    parser.add_argument("-l", "--numLayers", type=int, default=4, help="Number of stacked Encoder/Decoder layers (N)") # Adjusted default
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--numHeads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dff", type=int, default=1024, help="Dimension of feed-forward network (e.g., d_model * 4)") # d_model * 4 = 256 * 4 = 1024

    parser.add_argument("-D", "--nDemo", type=int, default=0, help="Number of test samples to predict. 0 for full validation.")
    parser.add_argument("-T", "--trainData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_train.tsv", help="Training data TSV")
    parser.add_argument("-t", "--testData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_valid.tsv", help="Test data TSV")
    parser.add_argument("-rp", "--resPath", type=str, default=os.getcwd(), help="Path for results and models")
    args = parser.parse_args()

    run = wandb.init(project="ncd_reasoning_tf_Transformer", config=vars(args)) # Updated project name
    cfg = run.config
    
    project_name = run.project or "wandb_project"
    sweep_id     = run.sweep_id
    run_id       = run.id
    run_parts = [project_name]
    if sweep_id is not None: run_parts.append(f"sweep-{sweep_id}") # Clarify sweep folder
    run_parts.append(f"run-{run_id}")
    run_root = os.path.join(os.path.normpath(getattr(cfg, "resPath", args.resPath)), *run_parts) + os.sep
    os.makedirs(run_root, exist_ok=True)    

    sequence_length = getattr(cfg, "seqLen", args.seqLen)
    max_features    = getattr(cfg, "nFeatures", args.nFeatures)
    batch_size      = getattr(cfg, "batchSize", args.batchSize)
    n_epochs        = getattr(cfg, "nEpochs", args.nEpochs)
    
    d_model         = getattr(cfg, "nSteps", args.nSteps) # nSteps is d_model
    if getattr(cfg, "embeddingDim", args.embeddingDim) != d_model:
        logging.warning(f"embeddingDim ({getattr(cfg, 'embeddingDim', args.embeddingDim)}) " \
                        f"is different from d_model (nSteps: {d_model}). Using d_model for embedding.")
    # For Transformer, embeddingDim must be d_model.

    num_layers      = getattr(cfg, "numLayers", args.numLayers)
    dropout_rate    = getattr(cfg, "dropout", args.dropout)
    num_heads       = getattr(cfg, "numHeads", args.numHeads)
    dff             = getattr(cfg, "dff", args.dff)
    maximum_position_encoding = 5000 # Fixed large value for positional encoding table

    training_data = getattr(cfg, "trainData", args.trainData)
    testing_data  = getattr(cfg, "testData",  args.testData)
    n_demo = args.nDemo # Use args.nDemo directly as it's not a typical hyperparameter for sweep

    gpus = tf.config.list_physical_devices('GPU')
    for dev in gpus: tf.config.experimental.set_memory_growth(dev, True)
    if gpus: mixed_precision.set_global_policy("mixed_float16")
    else: mixed_precision.set_global_policy("float32") # No mixed precision on CPU

    logging.info("Preparing train and test data")
    with open(training_data, 'r', encoding='utf-8') as f: train_text = f.readlines()
    with open(testing_data, 'r', encoding='utf-8') as f: val_text = f.readlines()
    train_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), train_text))
    val_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), val_text))

    train_in, train_out = zip(*train_pairs) if train_pairs else ([], [])
    test_in, test_out = zip(*val_pairs) if val_pairs else ([], [])
    train_in  = [str(s) for s in train_in]; train_out = [str(s) for s in train_out]
    test_in   = [str(s) for s in test_in];  test_out  = [str(s) for s in test_out]

    dataset = (tf.data.Dataset.from_tensor_slices((train_in, train_out))
               .shuffle(len(train_in), reshuffle_each_iteration=True) # Reshuffle each epoch
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_in, test_out))
                    .batch(batch_size, drop_remainder=True) # No shuffle for validation usually
                    .prefetch(tf.data.AUTOTUNE))

    input_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features, output_sequence_length=sequence_length,
        standardize=custom_standardization)
    output_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features, output_sequence_length=sequence_length + 1, # For [start]/[end]
        standardize=custom_standardization)
    
    # Adapt with all available text to build comprehensive vocabularies
    all_input_texts = [pair[0] for pair in train_pairs + val_pairs if pair]
    all_output_texts = [pair[1] for pair in train_pairs + val_pairs if pair] # Assuming pair[1] is string
                                                                                # If CS_LABELS, pair[1] is tuple (text, label)
    if CS_LABELS: # Handle tuple if CS_LABELS is True
        all_output_texts_actual = [item[0] for item in all_output_texts if isinstance(item, tuple)]
    else:
        all_output_texts_actual = all_output_texts

    if not all_input_texts : all_input_texts.append("") # Ensure not empty for adapt
    if not all_output_texts_actual : all_output_texts_actual.append("")

    logging.info("Adapting input text vectorizer")
    input_vectorizer.adapt(all_input_texts)
    logging.info("Adapting output text vectorizer")
    output_vectorizer.adapt(all_output_texts_actual)
    
    vectorizer_path = os.path.join(run_root, "vectorizer") + os.sep
    os.makedirs(vectorizer_path, exist_ok=True)
    save_vectorizer(input_vectorizer, f"{vectorizer_path}in_vect_model.keras")
    save_vectorizer(output_vectorizer, f"{vectorizer_path}out_vect_model.keras")
    logging.info(f"Saved text vectorizers to {vectorizer_path}")
    
    # Check vocab size after adapting
    actual_input_vocab_size = input_vectorizer.vocabulary_size()
    actual_output_vocab_size = output_vectorizer.vocabulary_size()
    logging.info(f"Actual input vocab size: {actual_input_vocab_size}, output: {actual_output_vocab_size}")


    train_translator = TrainTranslator(
        d_model=d_model, num_heads=num_heads, dff=dff, num_layers=num_layers,
        input_text_processor=input_vectorizer,
        output_text_processor=output_vectorizer,
        maximum_position_encoding=maximum_position_encoding,
        dropout=dropout_rate
    )
    
    # Optimizer with a learning rate schedule is common for Transformers
    # For simplicity, using Adam with default LR as in original script.
    # Consider tf.keras.optimizers.schedules.LearningRateSchedule for advanced use.
    train_translator.compile(
        optimizer=tf.keras.optimizers.Adam(), # Add learning_rate=... if using schedule
        loss=MaskedLoss(), 
        metrics=[keras.metrics.AUC(name="auroc")] # Keras will also track 'accuracy' from train/test_step
    )
    
    checkpoint_dir  = os.path.join(run_root, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5") # .weights.h5 for save_weights_only
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, monitor="val_loss", save_best_only=True)
    wandb_cb    = wandb.keras.WandbCallback(save_model=False, log_weights=False, log_gradients=False, monitor="val_loss")    
    # BatchLogs were for custom loop. Keras handles batch metrics for progress bar.
    # If specific batch logging to W&B is needed, it can be re-added.
    
    translator_for_logging = Translator(train_translator.encoder, train_translator.decoder, input_vectorizer, output_vectorizer)
    sample_sentence_for_log = val_pairs[0][0] if val_pairs else (train_pairs[0][0] if train_pairs else "default sample sentence")
    attn_cb = AttentionLogger(translator_for_logging, sample_sentence_for_log)
    overfit_cb = make_overfit_callback(total_epochs=n_epochs)

    callbacks_list = [cp_callback, wandb_cb, attn_cb, overfit_cb]
    
    logging.info("Training Transformer model...")
    history = train_translator.fit(dataset, validation_data=test_dataset, epochs=n_epochs, callbacks=callbacks_list)
    logging.info("Training completed successfully")

    # Load best weights for inference if save_best_only was used
    if os.path.exists(checkpoint_path) and cp_callback.save_best_only:
        logging.info(f"Loading best weights from {checkpoint_path} for final evaluation and inference.")
        train_translator.load_weights(checkpoint_path)
    
    out_dir = run_root # Results saved in the run-specific directory
    logging.info("Saving evaluation results...")
    rdf = pd.DataFrame(history.history)
    rdf.to_csv(f"{out_dir}history.csv", index_label="epoch") # Save epoch numbers
    if not rdf.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8)) # Adjusted figsize
        # Plot loss and val_loss
        loss_cols = [col for col in rdf.columns if 'loss' in col.lower()]
        if loss_cols: rdf[loss_cols].plot(ax=axes[0], title="Loss")
        # Plot accuracy and val_accuracy (and AUROC)
        metric_cols = [col for col in rdf.columns if 'loss' not in col.lower()]
        if metric_cols: rdf[metric_cols].plot(ax=axes[1], title="Metrics")
        plt.tight_layout()
        plt.savefig(f"{out_dir}history_plot.pdf")
        plt.close(fig)
    else:
        logging.warning("History dataframe is empty. Skipping plot generation.")


    if n_demo >= 0:
        inference_samples = list(val_pairs) # Use validation pairs for demo
        random.shuffle(inference_samples)
        if n_demo > 0:
            inference_samples = inference_samples[:n_demo]
        
        inp_, targ_ = zip(*inference_samples) if inference_samples else ([], [])
        results_list = []
        
        if inp_:
            logging.info(f"Performing inferences on {len(inp_)} samples...")
            # Create a new Translator instance with the potentially updated (best) weights
            inference_translator = Translator(train_translator.encoder, train_translator.decoder, input_vectorizer, output_vectorizer)
            
            num_sections = math.ceil(len(inp_) / batch_size)
            for i_chunk, chunk in enumerate(np.array_split(list(inp_), num_sections)):
                logging.info(f"Inferring chunk {i_chunk+1}/{num_sections}")
                # tf_translate expects explicit args now
                translation_output = inference_translator.tf_translate(
                    tf.constant(chunk), 
                    tf.constant(sequence_length, dtype=tf.int32), # max_length for translation
                    tf.constant(False, dtype=tf.bool),           # return_attention
                    tf.constant(0.0, dtype=tf.float32)           # temperature (greedy)
                )
                results_list.extend(translation_output['text'].numpy().astype(str))
            
            result_df = pd.DataFrame({'Input_Phrase': inp_, 'Predicted_Output': results_list, 'True_Output': targ_})
            result_df.to_csv(f"{out_dir}predictions.csv", index=False)
            print("\nSample Predictions:")
            print(result_df.head())
            logging.info(f"Prediction results written to {out_dir}predictions.csv")
        else:
            logging.warning("No inference samples to process (n_demo might be 0 and val_pairs empty, or n_demo was negative).")

    wandb.finish()

if __name__ == "__main__":
    main()
