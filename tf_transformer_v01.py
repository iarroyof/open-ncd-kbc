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
    
    Args:
        line (str): Input TSV line.
        start_token (str): Token to prepend to output phrases.
        end_token (str): Token to append to output phrases.
        pmid (bool): Whether to remove the first column (e.g., PMID).
        include_labels (bool): Include the label in the output.
        include_sent (bool): Include the sentence in the input.
        all_start_end (bool): Add start/end tokens to input as well.
    
    Returns:
        tuple: (input_phrase, output_phrase)
    """
    line = line.split('\t')
    if pmid:
        line.pop(0)  # Remove PMID column
    # Extract predicate and clean it
    pred = ' '.join(re.findall('[A-Z][a-z]*', line[1])).lower() or line[1]
    # Handle cases where the fifth column isn't a digit
    if not line[4].strip().isdigit() and not re.match(r'^-?\d+(?:\.\d+)$', line[4].strip()):
        complements = []
        i = 4
        while not line[i].isdigit():
            complements.append(line[i])
            line.pop(i)
        line[3] = " ".join([line[3]] + complements)
    # Construct sample
    sample = [line[0], pred, line[2], f"{start_token}{line[3]}{end_token}", float(line[4].strip())]
    # Adjust output based on flags
    if not include_labels:
        del sample[-1]
        sample_o = sample[-1]
    else:
        sample_o = tuple(sample[-2:])
    # Adjust input based on flags
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
    Ensure loading uses the new .keras filenames:
    # When loading vectorizers, use:
    input_vectorizer = load_vectorizer(f"{vectorizer_path}in_vect_model.keras")
    output_vectorizer = load_vectorizer(f"{vectorizer_path}out_vect_model.keras")
    """
    model = tf.keras.models.load_model(from_file)
    vocab = model.layers[0].get_vocabulary()
    config = model.layers[0].get_config()
    # In TF 2.10+, no need to delete output_mode
    vectorizer = TextVectorization.from_config(config)
    # Initialize vocabulary
    vectorizer.adapt(['Initializing vectorizer'])
    vectorizer.set_vocabulary(vocab)
    return vectorizer


def save_vectorizer(vectorizer, to_file):
    """Save a TextVectorization layer to disk."""
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer
    ])
    model.compile()
    model.save(to_file)


def parse_dataset_name(training_data):
    """Parse dataset name from the training data file path."""
    training_data = training_data.split(os.sep)[-1].lower()
    names_dic = {"NCD": 'ncd' in training_data, "GP": 'gp' in training_data, "CN": 'conceptnet' in training_data}
    return '-'.join(k for k, v in names_dic.items() if v)


class MaskedLoss(tf.keras.losses.Loss):
    """Custom loss function that masks padding tokens."""
    def __init__(self):
        super().__init__(name='masked_loss')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        # 1. Up‑cast logits to float32—recommended for stability when the
        #    model runs in mixed‑precision (policy "mixed_float16").
        y_pred = tf.cast(y_pred, tf.float32)

        # 2. Compute per‑token cross‑entropy.
        loss = self.loss(y_true, y_pred)                # (B, T) float32

        # 3. Create mask **in the same dtype** as `loss` to avoid the
        #    “type float32 vs float16” multiply error.
        mask = tf.cast(y_true != 0, loss.dtype)         # float32

        return tf.reduce_sum(loss * mask)

# --- Model Architecture ---

class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau attention mechanism for sequence-to-sequence models."""
    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.attention = layers.AdditiveAttention()

    def call(self, query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[tf.ones(tf.shape(query)[:-1], dtype=bool), mask],
            return_attention_scores=True
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))
        return context_vector, attention_weights
    
    # Add get_config method for serialization compatibility
    def get_config(self):
        config = super().get_config()
        return config

class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model))
        pe = tf.zeros((max_len, d_model))
        pe = pe + tf.sin(position * div_term)[None, :, :]
        pe = pe + tf.cos(position * div_term)[None, :, :]
        self.pe = tf.constant(pe)

    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1]]

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_units, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)  # Use `training` param
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # Use `training` param
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ffn_units, input_vocab_size, dropout_rate=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, ffn_units, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        mask = tf.math.not_equal(inputs, 0)[:, tf.newaxis, tf.newaxis, :]
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return x

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_units, dropout_rate=0.1):
        super().__init__()
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_units, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.layernorm3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, inputs, enc_output, training=False):
        tar_seq_len = tf.shape(inputs)[1]
        look_ahead_mask = tf.linalg.band_part(tf.ones((tar_seq_len, tar_seq_len)), -1, 0)
        dec_mask = tf.math.not_equal(inputs, 0)[:, tf.newaxis, tf.newaxis, :]
        dec_mask = tf.logical_and(dec_mask, tf.cast(look_ahead_mask, tf.bool)[tf.newaxis, tf.newaxis, :, :])

        attn1 = self.mha1(query=inputs, value=inputs, key=inputs, attention_mask=dec_mask)
        attn1 = self.dropout1(attn1, training=training)  # Use `training` param
        out1 = self.layernorm1(inputs + attn1)
    
        attn2 = self.mha2(query=out1, value=enc_output, key=enc_output, 
                         attention_mask=dec_mask[:, :, :, :tf.shape(enc_output)[1]])
        attn2 = self.dropout2(attn2, training=training)  # Use `training` param
        out2 = self.layernorm2(out1 + attn2)
    
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)  # Use `training` param
        return self.layernorm3(out2 + ffn_output)

class TransformerDecoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ffn_units, target_vocab_size, dropout_rate=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, ffn_units, dropout_rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, enc_output, training=False):
        tar_seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_output, training=training)
        return self.final_layer(x)

class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor, num_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=8,
            ffn_units=4 * embedding_dim,
            input_vocab_size=input_text_processor.vocabulary_size(),
            dropout_rate=dropout
        )
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=8,
            ffn_units=4 * embedding_dim,
            target_vocab_size=output_text_processor.vocabulary_size(),
            dropout_rate=dropout
        )
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.shape_checker = ShapeChecker()
        self.train_metric = keras.metrics.SparseCategoricalAccuracy()
        self.test_metric = keras.metrics.SparseCategoricalAccuracy()

    def _preprocess(self, input_text, target_text):
        self.shape_checker(input_text, ('batch',))
        self.shape_checker(target_text, ('batch',))
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0
        return input_tokens, input_mask, target_tokens, target_mask

    def _loop_step(self, new_tokens, input_mask, enc_output):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
        decoder_input = input_token
        dec_logits = self.decoder(decoder_input, enc_output)
        return target_token, dec_logits

    def _train_step(self, inputs):
        input_text, target_text = inputs
        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)
        max_t = tf.shape(target_tokens)[1]
        with tf.GradientTape() as tape:
            enc_output = self.encoder(input_tokens)
            total_loss = tf.constant(0.0, tf.float32)
            for t in tf.range(max_t - 1):
                new_tokens = target_tokens[:, t:t + 2]
                y_true, y_pred = self._loop_step(new_tokens, input_mask, enc_output)
                y_pred = tf.cast(y_pred, tf.float32)
                total_loss += self.loss(y_true, y_pred)
                mask = tf.cast(tf.squeeze(y_true, 1) != 0, tf.float32)
                self.train_metric.update_state(
                    tf.squeeze(y_true, 1),
                    tf.squeeze(y_pred, 1),
                    sample_weight=mask
                )
            average_loss = total_loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
            scaled_loss = self.optimizer.get_scaled_loss(average_loss)
        scaled_grads = tape.gradient(scaled_loss, self.trainable_variables)
        grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': average_loss, 'accuracy': self.train_metric.result()}

    def _test_step(self, inputs):
        input_text, target_text = inputs
        (input_tokens, input_mask,
         target_tokens, target_mask) = self._preprocess(input_text, target_text)
        max_t = tf.shape(target_tokens)[1]
        enc_output = self.encoder(input_tokens)
        total_loss = tf.constant(0.0, tf.float32)
        for t in tf.range(max_t - 1):
            new_tokens = target_tokens[:, t:t + 2]
            y_true, y_pred = self._loop_step(new_tokens, input_mask, enc_output)
            y_pred = tf.cast(y_pred, tf.float32)
            total_loss += self.loss(y_true, y_pred)
            self.test_metric.update_state(
                tf.squeeze(y_true, 1),
                tf.squeeze(y_pred, 1),
                sample_weight=tf.cast(tf.squeeze(y_true, 1) != 0, tf.float32)
            )
        average_loss = total_loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
        return {'loss': average_loss, 'accuracy': self.test_metric.result()}

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.string)
        )
    )
    def _tf_train_step(self, input_batch, target_batch):
        return self._train_step((input_batch, target_batch))

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.string)
        )
    )
    def _tf_test_step(self, input_batch, target_batch):
        return self._test_step((input_batch, target_batch))

    def train_step(self, data):
        input_batch, target_batch = data
        return self._tf_train_step(input_batch, target_batch)

    def test_step(self, data):
        input_batch, target_batch = data
        return self._tf_test_step(input_batch, target_batch)

    def get_config(self):
        return {
            "embedding_dim": self.encoder.d_model,
            "units": self.encoder.d_model,
            "num_layers": self.encoder.num_layers,
            "dropout": self.encoder.dropout_rate,
        }


class Translator(tf.Module):
    """Inference class for translating input sequences using a trained model."""
    def __init__(self, encoder, decoder, input_text_processor, output_text_processor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        
        # Updated StringLookup usage for TF 2.10+
        self.output_token_string_from_index = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='', invert=True)
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        
        token_mask_ids = index_from_string(['', '[UNK]', '[start]']).numpy()
        self.token_mask = np.zeros(index_from_string.vocabulary_size(), dtype=bool)
        self.token_mask[token_mask_ids] = True
        self.start_token = index_from_string(tf.constant('[start]'))
        self.end_token = index_from_string(tf.constant('[end]'))

    def tokens_to_text(self, result_tokens):
        shape_checker = ShapeChecker()
        shape_checker(result_tokens, ('batch', 't'))
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        return tf.strings.strip(result_text)

    def sample(self, logits, temperature):
        """
        Apply the vocabulary mask and (optionally) temperature sampling.
        The logits are up‑cast to float32 for numerical stability.
        Returns int64 ids shaped (B, 1).
        """
        # ── make dtypes consistent ────────────────────────────────────
        logits = tf.cast(logits, tf.float32)             # (B,1,V) fp32
        mask   = self.token_mask[tf.newaxis, tf.newaxis, :]  # (1,1,V) bool
        logits = tf.where(mask,
                          tf.constant(-np.inf, dtype=tf.float32),
                          logits)

        if temperature == 0.0:                           # greedy
            return tf.argmax(logits, axis=-1,
                             output_type=tf.int64)       # (B,1)

        # categorical requires 2‑D [batch, classes]
        logits = tf.squeeze(logits, axis=1)              # (B,V)
        return tf.random.categorical(logits / temperature,
                                     num_samples=1,
                                     dtype=tf.int64)     # (B,1)


    def translate(self, input_text, *, max_length=50, temperature=1.0):
        input_tokens = self.input_text_processor(input_text)
        enc_output = self.encoder(input_tokens, training=False)
        
        batch_size = tf.shape(input_text)[0]
        result_tokens = tf.fill([batch_size, 1], self.start_token)
        
        for _ in range(max_length):
            dec_logits = self.decoder(result_tokens, enc_output, training=False)
            last_logits = dec_logits[:, -1, :]
            new_tokens = self.sample(last_logits, temperature)
            result_tokens = tf.concat([result_tokens, new_tokens], axis=1)
            
            if tf.reduce_all(tf.equal(new_tokens, self.end_token)):
                break
                
        return {'text': self.tokens_to_text(result_tokens)}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        return self.translate(input_text)


class BatchLogs(tf.keras.callbacks.Callback):
    """Callback to log batch metrics during training."""
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])

# ──────────────────────────────────────────────────────────────────────────────
#  Callback: log one attention heat‑map per epoch
# ──────────────────────────────────────────────────────────────────────────────
class AttentionLogger(keras.callbacks.Callback):
    """
    Logs a heat‑map of the attention matrix for one fixed validation sample,
    *without* padded columns / rows and with the **real words** on the axes.
    """
    def __init__(self, translator, sample_sentence: str):
        """
        translator : instance of the `Translator` class above  
        sample_sentence : the source sentence that will be fed every epoch
        """
        super().__init__()
        self.translator = translator
        self.sample     = tf.constant([sample_sentence])

        # 1⃣  build two inverse look‑ups once – cheap & re‑usable
        self.src_lookup = tf.keras.layers.StringLookup(
            vocabulary=translator.input_text_processor.get_vocabulary(),
            mask_token='', invert=True)

        self.tgt_lookup = tf.keras.layers.StringLookup(
            vocabulary=translator.output_text_processor.get_vocabulary(),
            mask_token='', invert=True)

    # ------------------------------------------------------------------  
    def on_epoch_end(self, epoch, logs=None):
        """
        • runs the model  
        • trims pads  
        • converts token ids → strings  
        • logs a single W&B image
        """
        out   = self.translator.translate(self.sample, return_attention=True)
        attn  = tf.squeeze(out['attention'], 0)          # (dec_T, enc_T)

        # ---------- original token ids ---------------------------------
        src_ids = self.translator.input_text_processor(self.sample)[0].numpy()
        tgt_ids = self.translator.output_text_processor(out['text'])[0].numpy()

        # ---------- trim padding ---------------------------------------
        enc_len = int((src_ids != 0).sum())
        dec_len = int((tgt_ids != 0).sum())
        attn    = attn[:dec_len, :enc_len]
        src_ids = src_ids[:enc_len]
        tgt_ids = tgt_ids[:dec_len]

        # ---------- ids ➜ strings --------------------------------------
        src_words = self.src_lookup(src_ids).numpy().astype(str)
        tgt_words = self.tgt_lookup(tgt_ids).numpy().astype(str)

        # ---------- plot -----------------------------------------------
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(enc_len * .4 + 1, dec_len * .4 + 1))
        im = ax.imshow(attn.numpy(), aspect='auto', cmap='viridis')

        ax.set_xticks(range(enc_len)); ax.set_xticklabels(src_words,
                                                          rotation=90,
                                                          fontsize=8)
        ax.set_yticks(range(dec_len)); ax.set_yticklabels(tgt_words,
                                                          fontsize=8)
        ax.set_xlabel("Encoder tokens"); ax.set_ylabel("Decoder tokens")
        fig.colorbar(im, ax=ax, fraction=.046)
        fig.tight_layout()

        # ---------- log to W&B -----------------------------------------
        step = int(self.model.optimizer.iterations.numpy())   # plain int
        wandb.log({"attention_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)



# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Train a sequence-to-sequence model with attention.")
    parser.add_argument("-s", "--seqLen", type=int, default=50, help="Per-sample sequence length")
    parser.add_argument("-u", "--nSteps", type=int, default=1024, help="Number of hidden recurrent steps (units)")
    parser.add_argument("-f", "--nFeatures", type=int, default=15000, help="Maximum vocabulary size")
    parser.add_argument("-b", "--batchSize", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--nEpochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("-d", "--embeddingDim", type=int, default=1024, help="Word embedding dimensionality")
    parser.add_argument("-l", "--numLayers", type=int, default=1, help="Number of stacked GRU layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability for GRU layers")
    parser.add_argument("-D", "--nDemo", type=int, default=0, help="Number of test samples to predict. default is 0, so predict on the full validation data.")
    parser.add_argument("-T", "--trainData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_train.tsv", help="Training data TSV")
    parser.add_argument("-t", "--testData", type=str, default="data/ncd_conceptnet/ncd_conceptnet_valid.tsv", help="Test data TSV")
    parser.add_argument("-rp", "--resPath", type=str, default=os.getcwd(), help="Path for results and models")
    args = parser.parse_args()

    # ----------  W&B init  --------------------------------
    #   • uses argparse values as the initial config
    #   • allows sweep overrides automatically
    run = wandb.init(
            project="ncd_reasoning_tf_GRU",               # change to your project
            config=vars(args),                     # makes each flag a wandb.config key
        
    )
    # keep a short alias
    cfg = run.config
    # ────────────────────────────────────────────────────────────────
    #  NEW: derive “run root” = results_path/project[/sweep]/run
    # ────────────────────────────────────────────────────────────────
    project_name = run.project or "wandb_project"
    sweep_id     = run.sweep_id            # None if not in a sweep
    run_id       = run.id                  # always present

    run_parts = [project_name]
    if sweep_id is not None:
        run_parts.append(sweep_id)
    run_parts.append(run_id)

    run_root = os.path.join(
        os.path.normpath(getattr(cfg, "resPath", args.resPath)),
        *run_parts) + os.sep           # final “/” for convenience
    os.makedirs(run_root, exist_ok=True)    
    # replace *all* reads of argparse fields with cfg.*
# ── hyper‑parameters (prefer sweep‑supplied values, else CLI defaults) ─────────
    sequence_length = getattr(cfg, "seqLen",       args.seqLen)
    max_features    = getattr(cfg, "nFeatures",    args.nFeatures)
    batch_size      = getattr(cfg, "batchSize",    args.batchSize)
    n_epochs        = getattr(cfg, "nEpochs",      args.nEpochs)
    embedding_dim   = getattr(cfg, "embeddingDim", args.embeddingDim)
    units           = getattr(cfg, "nSteps",       args.nSteps)
    num_layers      = getattr(cfg, "numLayers",    args.numLayers)
    dropout_rate    = getattr(cfg, "dropout",      args.dropout)
    training_data = getattr(cfg, "trainData", args.trainData)
    testing_data  = getattr(cfg, "testData",  args.testData)
    results_path  = os.path.normpath(
                   getattr(cfg, "resPath",  args.resPath)) + os.sep
    n_demo = args.nDemo
    # --- GPU runtime init ---------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    for dev in gpus:                                 # make TF allocate as‑needed
        tf.config.experimental.set_memory_growth(dev, True)
    mixed_precision.set_global_policy("mixed_float16")   # enable Tensor‑Cores

    dataset_name = parse_dataset_name(training_data)

    # Load and prepare data
    logging.info("Preparing train and test data")
    with open(training_data) as f:
        train_text = f.readlines()
    with open(testing_data) as f:
        val_text = f.readlines()
    train_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), train_text))
    val_pairs = list(map(functools.partial(prepare_data, include_labels=CS_LABELS, all_start_end=True), val_text))

    # Create datasets
    train_in, train_out = zip(*train_pairs)
    test_in, test_out = zip(*val_pairs)
    # force both lists to be plain strings
    train_in  = [str(s) for s in train_in]
    train_out = [str(s) for s in train_out]
    dataset = (
        tf.data.Dataset.from_tensor_slices((train_in, train_out))
          .shuffle(len(train_in))
          .batch(batch_size, drop_remainder=True)        # better for GPU :contentReference[oaicite:2]{index=2}
          .prefetch(tf.data.AUTOTUNE)                    # overlap I/O‑compute :contentReference[oaicite:3]{index=3}
    )
    test_in  = [str(s) for s in test_in]
    test_out = [str(s) for s in test_out]

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_in, test_out))
          .shuffle(len(test_in))
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
    )
    # Initialize and train vectorizers
    # Updated TextVectorization usage for TF 2.10+
    input_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features, output_sequence_length=sequence_length,
        standardize=custom_standardization)
    output_vectorizer = TextVectorization(
        output_mode="int", max_tokens=max_features, output_sequence_length=sequence_length + 1,
        standardize=custom_standardization)
    
    train_in_texts = [pair[0] for pair in train_pairs]
    train_out_texts = [pair[1][0] if CS_LABELS else pair[1] for pair in train_pairs]
    logging.info("Training input text vectorizer")
    input_vectorizer.adapt(train_in_texts)
    logging.info("Training output text vectorizer")
    output_vectorizer.adapt(train_out_texts)
    
    # Create directory if it doesn't exist
    vectorizer_path = os.path.join(run_root, "vectorizer") + os.sep
    os.makedirs(vectorizer_path, exist_ok=True)
    save_vectorizer(input_vectorizer, f"{vectorizer_path}in_vect_model.keras")
    save_vectorizer(output_vectorizer, f"{vectorizer_path}out_vect_model.keras")
    logging.info(f"Saved text vectorizers to {vectorizer_path}")
    max_features = max(len(input_vectorizer.get_vocabulary()), len(output_vectorizer.get_vocabulary()))

    # Setup model and training
    checkpoint_dir  = os.path.join(run_root, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5")
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_dir = run_root
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    # default W&B callback logs loss/accuracy per step+epoch,
    # system metrics, configurable plots, … :contentReference[oaicite:3]{index=3}
    wandb_cb    = wandb.keras.WandbCallback(
        save_model       = False,          # we already use ModelCheckpoint
        log_weights      = False,
        log_gradients    = False,
        monitor          = "val_loss")    
    train_loss, train_accu = BatchLogs('loss'), BatchLogs('accuracy')

    transformer_config = {
        'num_layers': num_layers,
        'd_model': embedding_dim,
        'num_heads': 8,
        'ffn_units': 4 * embedding_dim,
        'dropout_rate': dropout_rate
    }
    
    # Initialize encoder/decoder with Transformer config
    train_translator = TrainTranslator(embedding_dim, units, input_vectorizer, output_vectorizer, num_layers, dropout_rate)
    # Using Adam with default learning rate for TF 2.10+ compatibility
    train_translator.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=MaskedLoss(), 
        metrics=[keras.metrics.AUC(from_logits=True, name="auroc")]
    )
    
    logging.info("Training neural reasoning model...")
    translator = Translator(train_translator.encoder, train_translator.decoder, input_vectorizer, output_vectorizer)
    # fixed example = first validation sentence
    sample_sentence = val_pairs[0][0] if val_pairs else train_pairs[0][0]
    attn_cb = AttentionLogger(translator, sample_sentence)
    overfit_cb = make_overfit_callback(total_epochs=n_epochs)
    history = train_translator.fit(dataset, validation_data=test_dataset, epochs=n_epochs,
                                   callbacks=[train_loss, train_accu, cp_callback, wandb_cb, attn_cb, overfit_cb])
    logging.info("Training completed successfully")
    # ── store one‑number‑per‑run so sweeps can plot them ───────────
    # Save training history
    logging.info("Saving evaluation results...")
    rdf = pd.DataFrame(history.history)
    rdf.to_csv(f"{out_dir}history.csv")
    fig, axes = plt.subplots(2, 1)
    rdf[sort_cols(rdf.columns)].iloc[:, :2].plot(ax=axes[0])
    rdf[sort_cols(rdf.columns)].iloc[:, 2:].plot(ax=axes[1])
    plt.savefig(f"{out_dir}history_plot.pdf")

    # Perform inferences n_demo = 0 for doing so for the full validation data.
    if n_demo >= 0:
        random.shuffle(val_pairs)
        if n_demo > 0:
            val_pairs = val_pairs[:n_demo]
        inp_, targ_ = zip(*val_pairs)
        results = []
        logging.info("Performing inferences using the trained model...")
        translator = Translator(train_translator.encoder, train_translator.decoder, input_vectorizer, output_vectorizer)
        num_sections = math.ceil(len(inp_) / batch_size) if inp_ else 0
        if num_sections:
            for chunk in np.array_split(list(inp_), num_sections):
                result = translator.tf_translate(tf.constant(chunk))['text'].numpy()
                results.append(result.tolist())
            result = sum(results, [])
            result_df = pd.DataFrame({'Subj_Pred': inp_, 'Obj': result, 'Obj_true': targ_})
            result_df.to_csv(f"{out_dir}predictions.csv")
            print(result_df)
            logging.info(f"Results written to {out_dir}predictions.csv")
        else:
            logging.warning("No inference samples to process.")

    wandb.finish()

if __name__ == "__main__":
    main()
