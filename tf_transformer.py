from __future__ import annotations

# ── standard library ─────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import logging
import random
import re
import string
import yaml
from typing import List, Tuple
import zipfile

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import mixed_precision
import wandb
import matplotlib.pyplot as plt
# ── sequence-level metrics ───────────────────────────────────────────────────
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. Hyper-parameters dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass(kw_only=True)
class HParams:
    seq_len: int = 30
    vocab_size: int = 15000
    model_dim: int = 512
    latent_dim: int = 2048
    heads: int = 8
    stacks: int = 1
    key_dim: int | None = None
    batch: int = 64
    epochs: int = 30
    train_path: str | None = None
    valid_path: str | None = None
    out_dir: str = "results"
    seed: int = 42
    attn_sample_indices: List[int] | None = None  # New field for attention logging

    def __post_init__(self):
        if self.key_dim is None:
            self.key_dim = self.model_dim // self.heads
        if self.attn_sample_indices is None:
            self.attn_sample_indices = []
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def to_dict(self):
        return asdict(self)

# ════════════════════════════════════════════════════════════════════════════
# 2. Data utilities
# ════════════════════════════════════════════════════════════════════════════

START, END = "[start]", "[end]"
STRIP = string.punctuation.translate({ord("["): None, ord("]"): None})

@keras.saving.register_keras_serializable()
def standardize(text: tf.Tensor) -> tf.Tensor:
    text = tf.strings.lower(text)
    return tf.strings.regex_replace(text, f"[{re.escape(STRIP)}]", "")

def parse_line(line: str) -> Tuple[str, str]:
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 5:
        raise ValueError("Each row needs ≥5 tab-separated fields")
    cols.pop(0)
    raw_pred = cols[1]
    pred = " ".join(re.findall(r"[A-Z][a-z]*", raw_pred)).lower() or raw_pred
    if not cols[4].isdigit() and not re.match(r"^-?\d+(?:.\d+)?$", cols[4]):
        extras = []
        while not cols[4].isdigit():
            extras.append(cols.pop(4))
        cols[3] = " ".join([cols[3], *extras])
    src = f"{pred} {cols[2]}"
    tgt = f"{START} {cols[3]} {END}"
    return src, tgt

def parse_src(line: str) -> str:
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 4:
        raise ValueError("Each row needs ≥4 tab-separated fields for prediction")
    cols.pop(0)
    raw_pred = cols[1]
    pred = " ".join(re.findall(r"[A-Z][a-z]*", raw_pred)).lower() or raw_pred
    return f"{pred} {cols[2]}"

# ════════════════════════════════════════════════════════════════════════════
# 3. Vectorizer helpers
# ════════════════════════════════════════════════════════════════════════════

def build_vectorizer(vocab: int, seq_len: int) -> TextVectorization:
    return TextVectorization(
        max_tokens=vocab,
        output_sequence_length=seq_len,
        standardize=standardize,
        output_mode="int",
    )

def save_tv(tv: TextVectorization, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = keras.Sequential([keras.Input(shape=(1,), dtype="string"), tv])
    model.save(path, save_format="keras")
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.testzip()
        LOGGER.info(f"Saved and verified {path}")
    except zipfile.BadZipFile as e:
        LOGGER.error(f"Failed to save {path}: {e}")
        raise

# ════════════════════════════════════════════════════════════════════════════
# 4. Transformer building blocks
# ════════════════════════════════════════════════════════════════════════════

class MyLayerNorm(layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = self.add_weight('gamma', shape=(dim,), initializer='ones', trainable=True)
        self.beta = self.add_weight('beta', shape=(dim,), initializer='zeros', trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        normed = (x - mean) * tf.math.rsqrt(var + self.eps)
        return normed * self.gamma + self.beta

class PosEmbed(layers.Layer):
    def __init__(self, max_len: int, vocab: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.vocab = vocab
        self.dim = dim
        self.tok = layers.Embedding(vocab, dim)
        self.pos = layers.Embedding(max_len, dim)
        self.idx = tf.range(max_len)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        length = tf.shape(x)[-1]
        return self.tok(x) + self.pos(self.idx[:length])

    def compute_mask(self, x: tf.Tensor, _=None) -> None:
        return None

class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None, training: bool = False) -> tf.Tensor:
        attn = self.mha(tf.cast(x, tf.float32), tf.cast(x, tf.float32), training=training)
        x = self.norm1(tf.cast(x, tf.float32) + tf.cast(attn, tf.float32))
        ffn_out = self.ffn(tf.cast(x, tf.float32), training=training)
        return self.norm2(tf.cast(x, tf.float32) + tf.cast(ffn_out, tf.float32))

class DecBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(heads, key_dim)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)
        self.norm3 = MyLayerNorm(dim)

    def call(self, y: tf.Tensor, enc_out: tf.Tensor, training: bool = False) -> tf.Tensor:
        self_attn = self.self_mha(tf.cast(y, tf.float32), tf.cast(y, tf.float32), training=training)
        y = self.norm1(tf.cast(y, tf.float32) + tf.cast(self_attn, tf.float32))
        cross_attn = self.cross_mha(tf.cast(y, tf.float32), tf.cast(enc_out, tf.float32), training=training)
        y = self.norm2(tf.cast(y, tf.float32) + tf.cast(cross_attn, tf.float32))
        ffn_out = self.ffn(tf.cast(y, tf.float32), training=training)
        return self.norm3(tf.cast(y, tf.float32) + tf.cast(ffn_out, tf.float32))

# ════════════════════════════════════════════════════════════════════════════
# 5. Model builder
# ════════════════════════════════════════════════════════════════════════════

def build_model(h: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")
    x = PosEmbed(h.seq_len, h.vocab_size, h.model_dim)(enc_in)
    for _ in range(h.stacks):
        x = EncBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim)(x)
    enc_out = x
    y = PosEmbed(h.seq_len + 1, h.vocab_size, h.model_dim)(dec_in)
    for _ in range(h.stacks):
        y = DecBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim)(y, enc_out)
    y = layers.Dropout(0.1)(y)
    out = layers.Dense(h.vocab_size, activation="softmax")(y)
    return keras.Model([enc_in, dec_in], out, name="transformer")

# ════════════════════════════════════════════════════════════════════════════
# 6. Dataset pipeline
# ════════════════════════════════════════════════════════════════════════════

INPUT_VECT: TextVectorization
OUTPUT_VECT: TextVectorization

def _fmt(src: tf.Tensor, tgt: tf.Tensor):
    src_tok = INPUT_VECT(src)
    tgt_tok = OUTPUT_VECT(tgt)
    return {"encoder_inputs": src_tok, "decoder_inputs": tgt_tok[:, :-1]}, tgt_tok[:, 1:]

def make_ds(pairs: List[Tuple[str, str]], h: HParams) -> tf.data.Dataset:
    s, t = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(s), list(t)))
    return ds.batch(h.batch).map(_fmt).prefetch(tf.data.AUTOTUNE)

# ════════════════════════════════════════════════════════════════════════════
# 7. Prediction function
# ════════════════════════════════════════════════════════════════════════════

def batch_predict(model, src_vect, max_len, start_token, end_token, batch_size=32, temperature=0.7):
    batch_size = min(batch_size, len(src_vect))
    predictions = []
    vocab = OUTPUT_VECT.get_vocabulary()
    
    for i in range(0, len(src_vect), batch_size):
        batch_src = src_vect[i:i + batch_size]
        batch_size_actual = len(batch_src)
        enc_inputs = tf.convert_to_tensor(batch_src, dtype=tf.int64)
        dec_inputs = tf.fill([batch_size_actual, 1], tf.cast(start_token, tf.int64))
        output = [[] for _ in range(batch_size_actual)]
        finished = tf.zeros(batch_size_actual, dtype=tf.bool)
        
        LOGGER.info(f"Processing batch {i//batch_size + 1}, size: {batch_size_actual}")
        
        for step in range(max_len):
            preds = model.predict([enc_inputs, dec_inputs], verbose=0)
            logits = preds[:, -1, :] / temperature
            next_tokens = tf.random.categorical(logits, num_samples=1, dtype=tf.int64)
            next_tokens = tf.squeeze(next_tokens, axis=-1)
            
            if step == 0:
                top_probs = tf.sort(preds[0, -1, :], direction='DESCENDING')[:5]
                top_ids = tf.argsort(preds[0, -1, :], direction='DESCENDING')[:5]
                LOGGER.info(f"Step {step}, Sample probs: {top_probs.numpy()}, Tokens: {[vocab[id] for id in top_ids.numpy()]}")
            
            for j in range(batch_size_actual):
                if not finished[j]:
                    token = next_tokens[j].numpy()
                    output[j].append(token)
                    if step == 0:
                        LOGGER.info(f"Sequence {i+j}, Step {step}, Token: {token}, Word: {vocab[token]}")
                    if token == end_token:
                        finished = tf.tensor_scatter_nd_update(finished, [[j]], [True])
            
            if tf.reduce_all(finished):
                LOGGER.info(f"Batch {i//batch_size + 1} finished early at step {step + 1}")
                break
                
            dec_inputs = tf.concat([dec_inputs, tf.expand_dims(next_tokens, -1)], axis=1)
        
        predictions.extend(output)
    
    return predictions

# ════════════════════════════════════════════════════════════════════════════
# 8. Attention logging callback
# ════════════════════════════════════════════════════════════════════════════

class AttentionLoggerCallback(keras.callbacks.Callback):
    def __init__(self, h, valid_pairs, input_vect, output_vect, log_samples=5):
        super().__init__()
        self.h = h
        self.valid_pairs = valid_pairs
        self.input_vect = input_vect
        self.output_vect = output_vect
        self.log_samples = log_samples
        self.start_token_id = self.output_vect([START])[0, 0].numpy()
        self.end_token_id = self.output_vect([END])[0, 0].numpy()

    def on_epoch_end(self, epoch, logs=None):
        # Select samples
        if self.h.attn_sample_indices:
            indices = [i for i in self.h.attn_sample_indices if 0 <= i < len(self.valid_pairs)]
            samples = [self.valid_pairs[i] for i in indices]
        else:
            samples = random.sample(self.valid_pairs, min(self.log_samples, len(self.valid_pairs)))
            indices = ["random"] * len(samples)
        
        src_texts, _ = zip(*samples)
        enc_inputs = self.input_vect(src_texts).numpy()

        # Generate predictions with temperature sampling
        batch_size = len(src_texts)
        enc_inputs_tensor = tf.convert_to_tensor(enc_inputs, dtype=tf.int64)
        dec_inputs = tf.fill([batch_size, 1], tf.cast(self.start_token_id, tf.int64))
        predictions = [[] for _ in range(batch_size)]
        finished = tf.zeros(batch_size, dtype=tf.bool)
        temperature = 0.7
        
        for step in range(self.h.seq_len):
            preds = self.model.predict([enc_inputs_tensor, dec_inputs], verbose=0)
            logits = preds[:, -1, :] / temperature
            next_tokens = tf.random.categorical(logits, num_samples=1, dtype=tf.int64)
            next_tokens = tf.squeeze(next_tokens, axis=-1)
            
            # Debug top probabilities
            if step < 3:  # Log for first 3 steps
                top_probs = tf.sort(preds[0, -1, :], direction='DESCENDING')[:5]
                top_ids = tf.argsort(preds[0, -1, :], direction='DESCENDING')[:5]
                vocab = self.output_vect.get_vocabulary()
                LOGGER.info(f"Sample 0, Step {step}, Top probs: {top_probs.numpy()}, Tokens: {[vocab[id] for id in top_ids.numpy()]}")
            
            for j in range(batch_size):
                if not finished[j]:
                    token = next_tokens[j].numpy()
                    predictions[j].append(token)
                    if token == self.end_token_id:
                        finished = tf.tensor_scatter_nd_update(finished, [[j]], [True])
            
            if tf.reduce_all(finished):
                break
                
            dec_inputs = tf.concat([dec_inputs, tf.expand_dims(next_tokens, -1)], axis=1)

        # Convert predictions to numpy arrays for attention model
        pred_inputs = [np.array(pred[:self.h.seq_len], dtype=np.int64) for pred in predictions]
        pred_inputs = tf.keras.preprocessing.sequence.pad_sequences(pred_inputs, maxlen=self.h.seq_len + 1, padding='post', value=0)

        # Build temporary model to output cross-attention scores
        enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
        dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")
        x = PosEmbed(self.h.seq_len, self.h.vocab_size, self.h.model_dim)(enc_in)
        for _ in range(self.h.stacks):
            x = EncBlock(self.h.model_dim, self.h.latent_dim, self.h.heads, self.h.key_dim)(x)
        enc_out = x
        y = PosEmbed(self.h.seq_len + 1, self.h.vocab_size, self.h.model_dim)(dec_in)
        cross_attn_outputs = []
        for _ in range(self.h.stacks):
            dec_block = DecBlock(self.h.model_dim, self.h.latent_dim, self.h.heads, self.h.key_dim)
            y = dec_block(y, enc_out)
            _, cross_attn = dec_block.cross_mha(
                tf.cast(y, tf.float32), tf.cast(enc_out, tf.float32), return_attention_scores=True
            )
            cross_attn_outputs.append(cross_attn)
        y = layers.Dropout(0.1)(y)
        out = layers.Dense(self.h.vocab_size, activation="softmax")(y)
        attn_model = keras.Model([enc_in, dec_in], [out] + cross_attn_outputs)
        attn_model.set_weights(self.model.get_weights())

        # Log heatmaps for each sample
        for sample_idx in range(len(src_texts)):
            # Trim source to non-padding tokens
            src_valid = np.where(enc_inputs[sample_idx] != 0)[0]
            src_end = src_valid[-1] + 1 if src_valid.size > 0 else 1
            src_tokens = [self.input_vect.get_vocabulary()[int(token)] for token in enc_inputs[sample_idx][:src_end]]

            # Trim predicted sequence to [end] token
            pred_tokens = predictions[sample_idx]
            tgt_end = len(pred_tokens)
            for i, token in enumerate(pred_tokens):
                if token == self.end_token_id:
                    tgt_end = i + 1
                    break
            tgt_tokens = [self.output_vect.get_vocabulary()[token] for token in pred_tokens[:tgt_end]]

            # Compute attention scores for the predicted sequence
            enc_input_single = enc_inputs[sample_idx:sample_idx+1]
            dec_input_single = pred_inputs[sample_idx:sample_idx+1]
            outputs = attn_model.predict([enc_input_single, dec_input_single], verbose=0)
            attn_scores = outputs[1:]  # Skip output logits, take attention scores

            # Log full predicted sequence for debugging
            pred_text = " ".join([self.output_vect.get_vocabulary()[token] for token in pred_tokens[:tgt_end]])
            LOGGER.info(f"Sample {sample_idx} (Index {indices[sample_idx]}): Predicted sequence: {pred_text}")

            # Log heatmaps for each layer
            for layer_idx, scores in enumerate(attn_scores):
                attn_matrix = scores[0, 0][:tgt_end, :src_end]  # Trim to valid spans
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(attn_matrix, cmap='viridis')
                ax.set_xticks(range(len(src_tokens)))
                ax.set_yticks(range(len(tgt_tokens)))
                ax.set_xticklabels(src_tokens, rotation=90)
                ax.set_yticklabels(tgt_tokens)
                ax.set_title(f"Epoch {epoch} Layer {layer_idx} Head 0 Sample {sample_idx} (Index {indices[sample_idx]})")
                plt.colorbar(im)
                wandb.log({f"attention/epoch_{epoch}_layer_{layer_idx}_sample_{sample_idx}": wandb.Image(fig)})
                plt.close(fig)
        LOGGER.info(f"Logged attention matrices for epoch {epoch}")

# ════════════════════════════════════════════════════════════════════════════
# 9. Custom Loss with Label Smoothing
# ════════════════════════════════════════════════════════════════════════════

def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred, label_smoothing=0.1):
    # Convert y_true to one-hot encoding
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    # Apply label smoothing: reduce confidence in true labels and distribute to other classes
    y_true = y_true * (1 - label_smoothing) + (label_smoothing / tf.cast(tf.shape(y_pred)[-1], tf.float32))
    # Compute categorical crossentropy
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

# ════════════════════════════════════════════════════════════════════════════
# Padding-aware accuracy  &  id→text helper
# ════════════════════════════════════════════════════════════════════════════

def masked_accuracy(y_true, y_pred):
    """
    Token-level accuracy that ignores padding (token id 0).
    """
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)           # (B,T)
    pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int64)   # (B,T)
    matches = tf.cast(tf.equal(tf.cast(y_true, tf.int64), pred_ids), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)


def _ids_to_text(id_seqs, vocab, end_id):
    """Convert lists of ids → space-separated strings, stopping at <end>."""
    texts = []
    for seq in id_seqs:
        words = []
        for tok in seq:
            if tok == end_id:
                break
            words.append(vocab[tok])
        texts.append(" ".join(words))
    return texts

# ════════════════════════════════════════════════════════════════════════════
# Callback to compute sequence-level metrics on the validation set
# ════════════════════════════════════════════════════════════════════════════

class TextMetricsCallback(keras.callbacks.Callback):
    """
    At the end of every epoch, generate the full prediction for each
    validation sample (greedy decoding) and log ROUGE-L, BLEU-4, METEOR.
    """
    def __init__(self, hparams, valid_pairs, input_vect, output_vect, batch_size=32):
        super().__init__()
        self.h = hparams
        self.valid_pairs = valid_pairs
        self.input_vect = input_vect
        self.output_vect = output_vect
        self.batch_size = batch_size

        self.start_id = output_vect([START])[0, 0].numpy()
        self.end_id   = output_vect([END ])[0, 0].numpy()
        self.vocab = output_vect.get_vocabulary()

        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.smooth = SmoothingFunction().method4  # BLEU smoothing

    # ------------------------------------------------------------------ #
    def _predict_all(self):
        src_texts, ref_texts = zip(*self.valid_pairs)
        src_vec = self.input_vect(src_texts).numpy()
        preds = batch_predict(
            self.model,
            src_vec,
            self.h.seq_len,
            self.start_id,
            self.end_id,
            batch_size=self.batch_size,
            temperature=0.0,  # greedy decoding
        )
        hyp = _ids_to_text(preds, self.vocab, self.end_id)
        return list(hyp), list(ref_texts)

    # ------------------------------------------------------------------ #
    def on_epoch_end(self, epoch, logs=None):
        hyps, refs = self._predict_all()

        rougeL = float(np.mean(
            [self.rouge.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
        ))
        bleu4 = corpus_bleu(
            [[r.split()] for r in refs],
            [h.split() for h in hyps],
            smoothing_function=self.smooth,
        )
        meteor = float(np.mean([
            meteor_score([r.split()], h.split())  # pass token lists
                for h, r in zip(hyps, refs)
        ]))


        # add to Keras logs and W&B
        logs = logs or {}
        logs.update({"val_rougeL": rougeL, "val_bleu4": bleu4, "val_meteor": meteor})
        wandb.log({"epoch": epoch,
                   "val_rougeL": rougeL,
                   "val_bleu4": bleu4,
                   "val_meteor": meteor},
                  step=epoch)
        LOGGER.info(f"[epoch {epoch}] ROUGE-L={rougeL:.4f} | "
                    f"BLEU-4={bleu4:.4f} | METEOR={meteor:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# 10. Main execution
# ════════════════════════════════════════════════════════════════════════════

def main():
    global INPUT_VECT, OUTPUT_VECT
    LOGGER.info("Starting tf_transformer.py")

    parser = argparse.ArgumentParser(description="Train Transformer model with optional immediate evaluation and weight saving")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--train-path", type=str, default=None, help="Path to training data")
    parser.add_argument("--valid-path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--train", action="store_true", help="Enable training")
    parser.add_argument("--evaluate", action="store_true", help="Enable immediate evaluation after training")
    parser.add_argument("--save-weights", action="store_true", help="Save model weights to ckpt.weights.h5")
    parser.add_argument("--eval-path", type=str, default=None, help="Not used; kept for compatibility")
    parser.add_argument("--seq-len", type=int, default=30, help="Maximum sequence length")
    parser.add_argument("--vocab-size", type=int, default=15000, help="Vocabulary size")
    parser.add_argument("--model-dim", type=int, default=512, help="Model embedding dimension")
    parser.add_argument("--latent-dim", type=int, default=2048, help="Feed-forward network latent dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--stacks", type=int, default=1, help="Number of transformer stacks")
    parser.add_argument("--key-dim", type=int, default=None, help="Attention key dimension (defaults to model_dim/heads)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--attn-sample-indices", type=int, nargs="*", default=None, help="Validation sample indices for attention logging")
    args = parser.parse_args()
    
    if args.train and args.train_path is None:
        parser.error("--train-path is required when --train is specified")

    # Enable mixed precision
    mixed_precision.set_global_policy('mixed_float16')
    LOGGER.info("Mixed precision training enabled with 'mixed_float16' policy")

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None and k in HParams.__dataclass_fields__:
            config[k] = v

    h = HParams(**config)

    if not args.train:
        LOGGER.info("No action specified; use --train to train the model")
        return

    train_lines = Path(h.train_path).read_text().splitlines()
    valid_lines = Path(h.valid_path).read_text().splitlines()
    train_pairs = [parse_line(l) for l in train_lines]
    valid_pairs = [parse_line(l) for l in valid_lines]

    INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
    OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
    INPUT_VECT.adapt([s for s, _ in train_pairs])
    OUTPUT_VECT.adapt([t for _, t in train_pairs])

    LOGGER.info(f"Output vocabulary size: {len(OUTPUT_VECT.get_vocabulary())}")
    LOGGER.info(f"Output vocabulary sample: {OUTPUT_VECT.get_vocabulary()[:20]}")

    wandb_config = asdict(h)
    wandb.init(project="tf-transformer", config=wandb_config, save_code=True)

    run_out_dir = Path(h.out_dir) / wandb.run.project / (wandb.run.sweep_id or "nosweep") / wandb.run.id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = h.to_dict()
    config_dict['out_dir'] = str(run_out_dir)
    with open(run_out_dir / "config.yaml", 'w') as f:
        yaml.dump(config_dict, f)

    h.out_dir = str(run_out_dir)

    save_tv(INPUT_VECT, Path(h.out_dir) / "vectorizers" / "input.keras")
    save_tv(OUTPUT_VECT, Path(h.out_dir) / "vectorizers" / "output.keras")
    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

    train_ds = make_ds(train_pairs, h)
    valid_ds = make_ds(valid_pairs, h)

    model = build_model(h)
    optimizer = tf.keras.optimizers.Adam()
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=sparse_categorical_crossentropy_with_smoothing,
        metrics= [masked_accuracy], #["sparse_categorical_accuracy"],
    )

    model.build(input_shape=[(None, None), (None, None)])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
        AttentionLoggerCallback(h, valid_pairs, INPUT_VECT, OUTPUT_VECT, log_samples=5),
        TextMetricsCallback(h, valid_pairs, INPUT_VECT, OUTPUT_VECT, batch_size=h.batch),
        wandb.keras.WandbCallback(save_model=False),
    ]
    if args.save_weights:
        callbacks.append(keras.callbacks.ModelCheckpoint(
            Path(h.out_dir) / "ckpt.weights.h5", save_weights_only=True, verbose=1
        ))

    hist = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=h.epochs,
        callbacks=callbacks,
    )
    pd.DataFrame(hist.history).to_csv(Path(h.out_dir) / "history.csv", index=False)

    if args.evaluate:
        valid_lines = Path(h.valid_path).read_text().splitlines()
        src_list = [parse_src(l) for l in valid_lines]
        src_vect = INPUT_VECT(src_list).numpy()

        start_token = OUTPUT_VECT([START])[0, 0].numpy()
        end_token = OUTPUT_VECT([END])[0, 0].numpy()
        LOGGER.info(f"Start token: {start_token}, End token: {end_token}")

        predictions = batch_predict(model, src_vect, h.seq_len, start_token, end_token)

        vocab = OUTPUT_VECT.get_vocabulary()
        pred_texts = [" ".join([vocab[token] for token in pred if token != end_token]) for pred in predictions]
        # ---------- Sequence-level metrics on evaluation split ----------
        refs = [parse_line(l)[1] for l in valid_lines]
        rouge_eval = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rougeL = np.mean([
            rouge_eval.score(r, h)["rougeL"].fmeasure
            for h, r in zip(pred_texts, refs)
        ])
        bleu4 = corpus_bleu(
            [[r.split()] for r in refs],
            [h.split() for h in pred_texts],
            smoothing_function=SmoothingFunction().method4,
        )
        meteor = np.mean([meteor_score([r], h) for h, r in zip(pred_texts, refs)])

        LOGGER.info(f"Evaluation metrics  |  ROUGE-L={rougeL:.4f}  "
                    f"BLEU-4={bleu4:.4f}  METEOR={meteor:.4f}")

        with open(run_out_dir / "eval_metrics.yaml", "w") as f:
            yaml.dump({"rougeL": float(rougeL),
                       "bleu4":  float(bleu4),
                       "meteor": float(meteor)}, f)

        with open(run_out_dir / "predictions.txt", 'w') as f:
            for text in pred_texts:
                f.write(text + "\n")
        LOGGER.info(f"Saved predictions to: {run_out_dir / 'predictions.txt'}")

if __name__ == "__main__":
    main()
