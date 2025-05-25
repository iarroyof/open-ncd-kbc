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
import wandb

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

    def __post_init__(self):
        if self.key_dim is None:
            self.key_dim = self.model_dim // self.heads
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
        y = self.norm1(
            tf.cast(y, tf.float32) +
            tf.cast(self.self_mha(tf.cast(y, tf.float32), tf.cast(y, tf.float32), training=training), tf.float32)
        )
        y = self.norm2(
            tf.cast(y, tf.float32) +
            tf.cast(self.cross_mha(tf.cast(y, tf.float32), tf.cast(enc_out, tf.float32), training=training), tf.float32)
        )
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

def batch_predict(model, src_vect, max_len, start_token, end_token, batch_size=32):
    batch_size = min(batch_size, len(src_vect))
    predictions = []
    vocab = OUTPUT_VECT.get_vocabulary()  # For debugging
    
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
            if step == 0:  # Debug first step
                top_probs = tf.sort(preds[0, -1, :], direction='DESCENDING')[:5]
                top_ids = tf.argsort(preds[0, -1, :], direction='DESCENDING')[:5]
                LOGGER.info(f"Step {step}, Sample probs: {top_probs.numpy()}, Tokens: {[vocab[id] for id in top_ids.numpy()]}")
            
            next_tokens = tf.argmax(preds[:, -1, :], axis=-1, output_type=tf.int64)
            
            for j in range(batch_size_actual):
                if not finished[j]:
                    token = next_tokens[j].numpy()
                    output[j].append(token)
                    if step == 0:  # Debug first token
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
# 8. Main execution
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
    args = parser.parse_args()
    
    if args.train and args.train_path is None:
        parser.error("--train-path is required when --train is specified")

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

    # Debug vocabulary
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
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
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
        valid_lines = Path(h.valid_path).read_text().splitlines()
        src_list = [parse_src(l) for l in valid_lines]
        src_vect = INPUT_VECT(src_list).numpy()

        start_token = OUTPUT_VECT([START])[0, 0].numpy()
        end_token = OUTPUT_VECT([END])[0, 0].numpy()
        LOGGER.info(f"Start token: {start_token}, End token: {end_token}")

        predictions = batch_predict(model, src_vect, h.seq_len, start_token, end_token)

        vocab = OUTPUT_VECT.get_vocabulary()
        pred_texts = [" ".join([vocab[token] for token in pred if token != end_token]) for pred in predictions]

        with open(run_out_dir / "predictions.txt", 'w') as f:
            for text in pred_texts:
                f.write(text + "\n")
        LOGGER.info(f"Saved predictions to {run_out_dir / 'predictions.txt'}")

if __name__ == "__main__":
    main()
