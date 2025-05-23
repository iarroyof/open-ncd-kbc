"""Transformer‑based text‑to‑text generation (TensorFlow 2.10+)
Minimal, self‑contained script: load TSV → train / evaluate Transformer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse, logging, random, re, string
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

# ───── HYPER‑PARAMS ─────────────────────────────────────────────────────────
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
        if self.key_dim is None:
            self.key_dim = self.model_dim // self.heads
        random.seed(self.seed); np.random.seed(self.seed); tf.random.set_seed(self.seed)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "vectorizers").mkdir(exist_ok=True)

START, END = "[start]", "[end]"
STRIP = string.punctuation.replace("[", "").replace("]", "")

# ───── DATA UTILS ───────────────────────────────────────────────────────────

def _std():
    return lambda s: tf.strings.regex_replace(tf.strings.lower(s), f"[{re.escape(STRIP)}]", "")


def parse_line(line: str) -> Tuple[str, str]:
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 5:
        raise ValueError("Bad TSV row")
    cols.pop(0)
    pred = " ".join(re.findall(r"[A-Z][a-z]*", cols[1])).lower() or cols[1]
    if not cols[4].isdigit() and not re.match(r"^-?\d+(?:\.\d+)?$", cols[4]):
        extra = []
        while not cols[4].isdigit():
            extra.append(cols.pop(4))
        cols[3] = " ".join([cols[3], *extra])
    src = f"{pred} {cols[2]}"
    tgt = f"{START} {cols[3]} {END}"
    return src, tgt

# ───── VECTORIZERS ─────────────────────────────────────────────────────────

def make_vect(vocab: int, seq: int) -> TextVectorization:
    return TextVectorization(max_tokens=vocab, output_sequence_length=seq, standardize=_std(), output_mode="int")


def save_vect(tv: TextVectorization, path: Path):
    keras.Sequential([keras.Input(shape=(1,), dtype="string"), tv]).save(path, save_format="keras")


def load_vect(path: Path) -> TextVectorization:
    m = keras.models.load_model(path)
    v_old: TextVectorization = m.layers[1]
    cfg, voc = v_old.get_config(), v_old.get_vocabulary()
    v_new = TextVectorization.from_config(cfg); v_new.adapt(["init"]); v_new.set_vocabulary(voc)
    return v_new

# ───── MODEL BLOCKS ────────────────────────────────────────────────────────
class PosEmbed(layers.Layer):
    def __init__(self, seq: int, vocab: int, dim: int):
        super().__init__(); self.tok = layers.Embedding(vocab, dim); self.pos = layers.Embedding(seq, dim)
        self.idx = tf.range(seq)
    def call(self, x):
        return self.tok(x) + self.pos(self.idx[: tf.shape(x)[-1]])
    def compute_mask(self, x, _=None):
        return tf.not_equal(x, 0)

class Enc(layers.Layer):
    def __init__(self, dim, lat, heads, k):
        super().__init__(); self.m = layers.MultiHeadAttention(heads, k); self.f = keras.Sequential([layers.Dense(lat, 'relu'), layers.Dense(dim)])
        self.n1, self.n2 = layers.LayerNormalization(), layers.LayerNormalization()
    def call(self, x, mask=None, training=None):
        x = self.n1(x + self.m(x, x, attention_mask=mask, training=training))
        return self.n2(x + self.f(x, training=training))

class Dec(layers.Layer):
    def __init__(self, dim, lat, heads, k):
        super().__init__(); self.s = layers.MultiHeadAttention(heads, k); self.c = layers.MultiHeadAttention(heads, k)
        self.f = keras.Sequential([layers.Dense(lat, 'relu'), layers.Dense(dim)]); self.n1, self.n2, self.n3 = [layers.LayerNormalization() for _ in range(3)]
    def _causal(self, t):
        return tf.linalg.band_part(tf.ones((t, t)), -1, 0)[None, None]
    def call(self, y, enc, y_m=None, e_m=None, training=None):
        t = tf.shape(y)[1]; c = self._causal(t)
        if y_m is not None:
            y_m = tf.cast(y_m[:, None, None, :], tf.int32); c = tf.minimum(c, y_m)
        y = self.n1(y + self.s(y, y, attention_mask=c, training=training))
        if e_m is not None:
            e_m = tf.cast(e_m[:, None, None, :], tf.int32)
        y = self.n2(y + self.c(y, enc, attention_mask=e_m, training=training))
        return self.n3(y + self.f(y, training=training))

# ───── BUILD MODEL ─────────────────────────────────────────────────────────

def build_model(h: HParams) -> keras.Model:
    ei, di = keras.Input((None,), dtype="int64"), keras.Input((None,), dtype="int64")
    x = PosEmbed(h.seq_len, h.vocab_size, h.model_dim)(ei)
    for _ in range(h.stacks):
        x = Enc(h.model_dim, h.latent_dim, h.heads, h.key_dim)(x)
    y = PosEmbed(h.seq_len + 1, h.vocab_size, h.model_dim)(di)
    for _ in range(h.stacks):
        y = Dec(h.model_dim, h.latent_dim, h.heads, h.key_dim)(y, x, y_mask=y._keras_mask, enc_mask=x._keras_mask)
    y = layers.Dropout(0.1)(y)
    out = layers.Dense(h.vocab_size, activation="softmax")(y)
    return keras.Model([ei, di], out)

# ───── DATASET PIPELINE ────────────────────────────────────────────────────
INPUT_V, OUTPUT_V = None, None

def _fmt(src: tf.Tensor, tgt: tf.Tensor):
    s = INPUT_V(src); t = OUTPUT_V(tgt); return {"input_1": s, "input_2": t[:, :-1]}, t[:, 1:]

def make_ds(pairs: List[Tuple[str, str]], h: HParams):
    s, t = zip(*pairs); ds = tf.data.Dataset.from_tensor_slices((list(s), list(t)))
    return ds.batch(h.batch).map(_fmt).prefetch(tf.data.AUTOTUNE)

# ───── CLI ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(); add = parser.add_argument
add("--train", action="store_true"); add("--evaluate", action="store_true")
add("--train-path", required=True); add("--valid-path", required=True)
add("--seq-len", type=int, default=30); add("--vocab-size", type=int, default=15000)
add("--batch", type=int, default=64); add("--epochs", type=int, default=30)
args = parser.parse_args()

hp = HParams(seq_len=args.seq_len, vocab_size=args.vocab_size, batch=args.batch, epochs=args.epochs,
             train_path=Path(args.train_path), valid_path=Path(args.valid_path))

train_pairs = [parse_line(l) for l in hp.train_path.read_text().splitlines()]
valid_pairs = [parse_line(l) for l in hp.valid_path.read_text().splitlines()]

INPUT_V = make_vect(hp.vocab_size, hp.seq_len); OUTPUT_V = make_vect(hp.vocab_size, hp.seq_len + 1)
INPUT_V.adapt([p[0] for p in train_pairs]); OUTPUT_V.adapt([p[1] for p in train_pairs])

save_vect(INPUT_V, hp.out_dir / "vectorizers" / "inp.keras"); save_vect(OUTPUT_V, hp.out_dir / "vectorizers" / "out.keras")

hp.vocab_size = max(len(INPUT_V.get_vocabulary()), len(OUTPUT_V.get_vocabulary()))
model = build_model(hp)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
model.summary()

train_ds, valid_ds = make_ds(train_pairs, hp), make_ds(valid_pairs, hp)
ckpt = keras.callbacks.ModelCheckpoint(hp.out_dir / "ckpt.h5", save_weights_only=True, verbose=1)
if args.train:
    model.fit(train_ds, validation_data=valid_ds, epochs=hp.epochs, callbacks=[ckpt])
if args.evaluate:
    loss, acc = model.evaluate(valid_ds, verbose=0); LOGGER.info("val_loss %.4f | val_acc %.4f", loss, acc)
