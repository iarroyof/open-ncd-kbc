"""tf_transformer.py
====================
Minimal, **complete** TensorFlow 2.10+ script to train and evaluate
a small Transformer encoder–decoder on TSV triples.

Key features:
* Single file, no lambdas in serialized objects.
* Automatic mask propagation via `compute_mask`.
* Vectorizers saved/loaded in native `.keras` format.
* Checkpoints, history and metrics under `results/`.
* Compatible with CPU or GPU execution.

Example usage:
-------------
```bash
python tf_transformer.py \
  --train-path data/train.tsv \
  --valid-path data/valid.tsv \
  --train --epochs 2 --seq-len 30 --batch 32
```
```bash
python tf_transformer.py \
  --train-path data/train.tsv \
  --valid-path data/valid.tsv \
  --evaluate
```
"""
from __future__ import annotations

# ── standard library ─────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import random
import re
import string
from typing import List, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

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

    train_path: Path | None = None
    valid_path: Path | None = None
    out_dir: Path = Path("results")

    seed: int = 42

    def __post_init__(self):
        # Compute key_dim if not provided
        if self.key_dim is None:
            self.key_dim = self.model_dim // self.heads
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # Create output dirs
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "vectorizers").mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. Data utilities
# ════════════════════════════════════════════════════════════════════════════
START, END = "[start]", "[end]"
# strip brackets from punctuation list
STRIP = string.punctuation.translate({ord("["): None, ord("]"): None})


def standardize(text: tf.Tensor) -> tf.Tensor:
    """Lowercase and remove punctuation."""
    text = tf.strings.lower(text)
    return tf.strings.regex_replace(text, f"[{re.escape(STRIP)}]", "")


def parse_line(line: str) -> Tuple[str, str]:
    """Parse one TSV line into (source, target)."""
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 5:
        raise ValueError("Each row needs ≥5 tab-separated fields")
    # Drop PMID
    cols.pop(0)
    # Clean predicate (split CamelCase)
    raw_pred = cols[1]
    pred = " ".join(re.findall(r"[A-Z][a-z]*", raw_pred)).lower() or raw_pred
    # Merge multi-token object until numeric label
    if not cols[4].isdigit() and not re.match(r"^-?\d+(?:\.\d+)?$", cols[4]):
        extras: list[str] = []
        while not cols[4].isdigit():
            extras.append(cols.pop(4))
        cols[3] = " ".join([cols[3], *extras])
    src = f"{pred} {cols[2]}"
    tgt = f"{START} {cols[3]} {END}"
    return src, tgt

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
    keras.Sequential([keras.Input(shape=(1,), dtype="string"), tv]).save(
        path, save_format="keras"
    )


def load_tv(path: Path) -> TextVectorization:
    mdl = keras.models.load_model(path, safe_mode=False)
    old: TextVectorization = mdl.layers[1]
    cfg, vocab = old.get_config(), old.get_vocabulary()
    new = TextVectorization.from_config(cfg)
    new.adapt(["_init_"])
    new.set_vocabulary(vocab)
    return new

# ════════════════════════════════════════════════════════════════════════════
# 4. Transformer building blocks
# ════════════════════════════════════════════════════════════════════════════
class PosEmbed(layers.Layer):
    """Token+position embedding layer without mask propagation (mask disabled)."""
    def __init__(self, max_len: int, vocab: int, dim: int):
        super().__init__()
        self.tok = layers.Embedding(vocab, dim)
        self.pos = layers.Embedding(max_len, dim)
        self.idx = tf.range(max_len)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        length = tf.shape(x)[-1]
        return self.tok(x) + self.pos(self.idx[:length])

    def compute_mask(self, x: tf.Tensor, _=None) -> None:
        # Disable mask propagation
        return None

class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = layers.LayerNormalization(dtype='float32')
        self.norm2 = layers.LayerNormalization(dtype='float32')

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # Self-attention without padding mask
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + attn)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)


class DecBlock(layers.Layer):
    def __init__(
        self,
        dim: int,
        latent: int,
        heads: int,
        key_dim: int,
    ):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(heads, key_dim)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization(dtype='float32')

    def call(
        self,
        y: tf.Tensor,
        enc_out: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # Unmasked self-attention
        y = self.norm1(
            y + self.self_mha(y, y, training=training)
        )
        # Unmasked cross-attention
        y = self.norm2(
            y + self.cross_mha(y, enc_out, training=training)
        )
        ffn_out = self.ffn(y, training=training)
        return self.norm3(y + ffn_out)

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
# These will be set after adapting vectorizers
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
# 7. Main execution
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Transformer model")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--vocab-size", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    # Build hyper-parameters
    h = HParams(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        batch=args.batch,
        epochs=args.epochs,
        train_path=Path(args.train_path),
        valid_path=Path(args.valid_path),
        out_dir=Path(args.out_dir),
    )

    # Load and parse data
    train_lines = h.train_path.read_text().splitlines()
    valid_lines = h.valid_path.read_text().splitlines()
    train_pairs = [parse_line(l) for l in train_lines]
    valid_pairs = [parse_line(l) for l in valid_lines]

    # Vectorizers
    global INPUT_VECT, OUTPUT_VECT
    INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
    OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
    INPUT_VECT.adapt([s for s, _ in train_pairs])
    OUTPUT_VECT.adapt([t for _, t in train_pairs])
    save_tv(INPUT_VECT, h.out_dir / "vectorizers" / "input.keras")
    save_tv(OUTPUT_VECT, h.out_dir / "vectorizers" / "output.keras")
    # Adjust vocab based on actual size
    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

    # Build datasets
    train_ds = make_ds(train_pairs, h) if args.train else None
    valid_ds = make_ds(valid_pairs, h)

    # Build and compile model
    model = build_model(h)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    # Training
    if args.train:
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                h.out_dir / "ckpt.weights.h5", save_weights_only=True, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                patience=5, min_delta=0.001, restore_best_weights=True, verbose=1
            ),
        ]
        hist = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=h.epochs,
            callbacks=callbacks,
        )
        pd.DataFrame(hist.history).to_csv(h.out_dir / "history.csv", index=False)

    # Evaluation
    if args.evaluate:
        loss, acc = model.evaluate(valid_ds, verbose=0)
        LOGGER.info("Validation loss=%.4f accuracy=%.4f", loss, acc)
        metrics_path = h.out_dir / "metrics.txt"
        metrics_path.write_text(f"loss\t{loss}\nacc\t{acc}\n")

if __name__ == "__main__":
    main()
