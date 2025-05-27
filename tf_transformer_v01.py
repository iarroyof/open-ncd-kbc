# LAtest working code with metrics correctly applied
from __future__ import annotations

# Standard library imports
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

# Third-party imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import mixed_precision
import wandb
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Hyper-parameters dataclass
@dataclass(kw_only=True)
class HParams:
    seq_len: int = 30
    vocab_size: int = 15000
    model_dim: int = 512
    latent_dim: int = 2048
    heads: int = 8
    stacks: int = 1
    key_dim: int | None = None
    dropout: float = 0.1
    batch: int = 64
    epochs: int = 30
    train_path: str | None = None
    valid_path: str | None = None
    out_dir: str = "results"
    seed: int = 42
    attn_sample_indices: List[int] | None = None

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

# Data utilities
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

# Vectorizer helpers
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

# Transformer building blocks
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
    def __init__(self, dim: int, latent: int,
                 heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.mha_drop = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)

    def call(self, x: tf.Tensor,
             mask: tf.Tensor | None = None,
             training: bool = False) -> tf.Tensor:
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.mha_drop(attn, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.ffn_drop(ffn_out, training=training))

class DecBlock(layers.Layer):
    def __init__(self, dim: int, latent: int,
                 heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.self_drop = layers.Dropout(dropout)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout)
        self.cross_drop = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)
        self.norm3 = MyLayerNorm(dim)

    def call(self, y: tf.Tensor, enc_out: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        self_attn = self.self_mha(y, y, training=training)
        y = self.norm1(y + self.self_drop(self_attn, training=training))
        cross_attn = self.cross_mha(y, enc_out, training=training)
        y = self.norm2(y + self.cross_drop(cross_attn, training=training))
        ffn_out = self.ffn(y, training=training)
        return self.norm3(y + self.ffn_drop(ffn_out, training=training))

# Model builder
def build_model(h: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")
    x = PosEmbed(h.seq_len, h.vocab_size, h.model_dim)(enc_in)
    for _ in range(h.stacks):
        x = EncBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, dropout=h.dropout)(x)
    enc_out = x
    y = PosEmbed(h.seq_len + 1, h.vocab_size, h.model_dim)(dec_in)
    for _ in range(h.stacks):
        y = DecBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, dropout=h.dropout)(y, enc_out)
    y = layers.Dropout(h.dropout)(y)
    out = layers.Dense(h.vocab_size, activation="softmax")(y)
    return keras.Model([enc_in, dec_in], out, name="transformer")

# Custom Masked Loss (adapted from LSTM script)
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name='masked_loss')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

# Translator Class (adapted from LSTM script for Transformer)
class Translator(tf.Module):
    def __init__(self, model, input_text_processor, output_text_processor):
        super().__init__()
        self.model = model
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
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
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        return tf.strings.strip(result_text)

    def sample(self, logits, temperature):
        logits = tf.cast(logits, tf.float32)
        mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(mask, tf.constant(-np.inf, dtype=tf.float32), logits)
        if temperature == 0.0:
            return tf.argmax(logits, axis=-1, output_type=tf.int64)
        logits = tf.squeeze(logits, axis=1)
        return tf.random.categorical(logits / temperature, num_samples=1, dtype=tf.int64)

    def translate(self, input_text, max_length=50, temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_inputs = input_tokens
        dec_inputs = tf.fill([batch_size, 1], self.start_token)
        result_tokens = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        
        for _ in range(max_length):
            preds = self.model([enc_inputs, dec_inputs], training=False)
            logits = preds[:, -1, :]
            new_tokens = self.sample(logits, temperature)
            done |= (new_tokens == self.end_token)
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            result_tokens.append(new_tokens)
            dec_inputs = tf.concat([dec_inputs, new_tokens], axis=-1)
            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)
        return {'text': result_text}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tf_translate(self, input_text):
        return self.translate(input_text)

# Dataset pipeline
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

# Padding-aware accuracy
def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
    matches = tf.cast(tf.equal(tf.cast(y_true, tf.int64), pred_ids), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

# Main execution
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
    parser.add_argument("--dropout", type=float, default=None, help="Global dropout rate used throughout the Transformer")

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
        loss=MaskedLoss(),
        metrics=[masked_accuracy],
    )

    model.build(input_shape=[(None, None), (None, None)])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
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
        translator = Translator(model, INPUT_VECT, OUTPUT_VECT)
        inp_, targ_ = zip(*valid_pairs)
        results = translator.tf_translate(tf.constant(list(inp_)))['text'].numpy().tolist()
        result_df = pd.DataFrame({'Subj_Pred': inp_, 'Obj': results, 'Obj_true': targ_})
        result_df.to_csv(run_out_dir / "predictions.csv", index=False)
        LOGGER.info(f"Saved predictions to: {run_out_dir / 'predictions.csv'}")

if __name__ == "__main__":
    main()
