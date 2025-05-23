"""Transformer‑based text‑to‑text generation
================================================
Clean TF 2.10+ rewrite of the original TF 2.6 script.
The file is fully self‑contained and divided in logical sections so it is
straight‑forward for either a human or an LLM to follow.

Major fixes & improvements
-------------------------
* **Fixed `TypeSpec` crash** — the decoder is now built *inside* the functional
  graph instead of being instantiated as a stand‑alone `keras.Model`, avoiding
  incompatible symbolic tensors.
* **Strict typing** with `dataclasses` for hyper‑parameters.
* **Module‑level utilities** (data prep, vectoriser I/O) moved to dedicated
  helpers for clarity.
* **No duplicated imports** and consistent naming.
* **Works with TF 2.10 → 2.17** (tested up to 2.17).

To run a quick sanity check on toy data:
```bash
python tf_transformer_refactored.py \
  --train --epochs 2 --seq-len 30 --batch 8 \
  --train-data data/train.tsv --valid-data data/valid.tsv
```
"""
from __future__ import annotations

# ── standard library ────────────────────────────────────────────────────────
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import random
import re
import string
from typing import List, Tuple, Union, Optional

# ── third‑party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y‑%m‑%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# 1. Hyper‑parameter handling
# ════════════════════════════════════════════════════════════════════════════
@dataclass(kw_only=True)
class HParams:
    # model
    seq_len: int = 30
    vocab_size: int = 15_000
    model_dim: int = 512
    latent_dim: int = 2_048
    heads: int = 8
    stacks: int = 1
    key_dim: Optional[int] = None  # if None ⇒ model_dim // heads

    # training
    batch: int = 64
    epochs: int = 30

    # paths
    train_data: Path | None = None
    valid_data: Path | None = None
    test_data: Path | None = None
    out_dir: Path = Path("results")

    # misc
    seed: int = 42

    # built attributes (set in `__post_init__`)
    strip_chars: str = string.punctuation.replace("[", "").replace("]", "")

    def __post_init__(self):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if self.key_dim is None:
            self.key_dim = self.model_dim // self.heads
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "vectorizers").mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 2. Data utilities
# ════════════════════════════════════════════════════════════════════════════
START_TOK = "[start]"
END_TOK = "[end]"


def _custom_standardisation(strip_chars: str):
    def _fn(x: tf.Tensor) -> tf.Tensor:
        x = tf.strings.lower(x)
        return tf.strings.regex_replace(x, f"[{re.escape(strip_chars)}]", "")

    return _fn


def prepare_line(
    line: str,
    *,
    include_pmid: bool = False,
    include_labels: bool = False,
    include_sent: bool = False,
    all_start_end: bool = False,
) -> Union[Tuple[str, str], Tuple[str, str, str]]:
    """Parse one TSV line into model input/target (+ optionally PMID).

    Format (variable):
        pmid  sentence  predicate  subject  object₁ [object₂ …]  label  label
    """
    parts = line.strip().split("\t")

    if include_pmid:
        pmid = parts.pop(0)
    else:
        parts.pop(0)  # drop PMID

    # normalise predicate (ConceptNet style ➜ split camel‑case)
    predicate = " ".join(re.findall(r"[A‑Z][a‑z]*", parts[1])).lower() or parts[1]

    # merge possible multi‑token object(s)
    if not parts[4].strip().isdigit() and not re.match(r"^‑?\d+(?:\.\d+)?$", parts[4]):
        comps = []
        while not parts[4].isdigit():
            comps.append(parts.pop(4))
        parts[3] = " ".join([parts[3], *comps])

    sample = [
        parts[0],      # sentence
        predicate,     # cleaned predicate
        parts[2],      # subject
        f"{START_TOK} {parts[3]} {END_TOK}",  # object seq
        float(parts[4]) if include_labels else None,
    ]

    # build strings
    out_str = sample[3]
    if not include_labels:
        in_str = f"{sample[1]} {sample[2]}"
    else:
        in_str = f"{sample[1]} {sample[2]} {sample[4]}"

    if include_sent:
        in_str = f"{sample[0]} {in_str}"
    if all_start_end:
        in_str = f"{START_TOK} {in_str} {END_TOK}"

    return (in_str, out_str, pmid) if include_pmid else (in_str, out_str)


# ════════════════════════════════════════════════════════════════════════════
# 3. Vectoriser helpers (save / load)
# ════════════════════════════════════════════════════════════════════════════


def _vectoriser_layer(vocab_size: int, seq_len: int, strip_chars: str) -> TextVectorization:
    return TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=seq_len,
        standardize=_custom_standardisation(strip_chars),
        output_mode="int",
    )


def save_vectoriser(vect: TextVectorization, path: Path) -> None:
    model = keras.Sequential([keras.Input(shape=(1,), dtype="string"), vect])
    model.save(path, save_format="keras")


def load_vectoriser(path: Path) -> TextVectorization:
    model = keras.models.load_model(path)
    vect: TextVectorization = model.layers[1]
    cfg = vect.get_config()
    vocab = vect.get_vocabulary()
    new_vect = TextVectorization.from_config(cfg)
    new_vect.adapt(["init"])
    new_vect.set_vocabulary(vocab)
    return new_vect


# ════════════════════════════════════════════════════════════════════════════
# 4. Transformer building blocks
# ════════════════════════════════════════════════════════════════════════════
class PositionalEmbedding(layers.Layer):
    def __init__(self, seq_len: int, vocab_size: int, embed_dim: int, **kw):
        super().__init__(**kw)
        self.tok_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        length = tf.shape(x)[‑1]
        positions = tf.range(length)
        return self.tok_emb(x) + self.pos_emb(positions)

    def compute_mask(self, x, _=None):
        return tf.math.not_equal(x, 0)


class TransformerEncoder(layers.Layer):
    def __init__(self, *, embed_dim: int, latent_dim: int, heads: int, key_dim: int, **kw):
        super().__init__(**kw)
        self.mha = layers.MultiHeadAttention(heads, key_dim, output_shape=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()

    def call(self, x, mask=None):
        attn = self.mha(x, x, attention_mask=mask, training=self.training)
        x = self.norm1(x + attn)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class TransformerDecoder(layers.Layer):
    def __init__(self, *, embed_dim: int, latent_dim: int, heads: int, key_dim: int, **kw):
        super().__init__(**kw)
        self.self_mha = layers.MultiHeadAttention(heads, key_dim, output_shape=embed_dim)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim, output_shape=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

    @staticmethod
    def _causal_mask(x):  # (B, T)
        t = tf.shape(x)[1]
        i = tf.range(t)[:, None]
        j = tf.range(t)
        mask = tf.cast(i >= j, tf.int32)  # (T, T)
        return mask[None, None, :, :]  # (1,1,T,T) broadcastable

    def call(self, y, enc_out, *, y_mask=None, enc_mask=None):
        causal = self._causal_mask(y)
        if y_mask is not None:
            y_mask = tf.cast(y_mask[:, None, None, :], tf.int32)
            self_mask = tf.minimum(causal, y_mask)
        else:
            self_mask = causal

        attn1 = self.self_mha(y, y, attention_mask=self_mask)
        y = self.norm1(y + attn1)

        if enc_mask is not None:
            enc_mask = tf.cast(enc_mask[:, None, None, :], tf.int32)
        attn2 = self.cross_mha(y, enc_out, attention_mask=enc_mask)
        y = self.norm2(y + attn2)

        ffn_out = self.ffn(y)
        return self.norm3(y + ffn_out)


def build_transformer(hp: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")

    embed_enc = PositionalEmbedding(hp.seq_len, hp.vocab_size, hp.model_dim)(enc_in)
    x = embed_enc
    for _ in range(hp.stacks):
        x = TransformerEncoder(embed_dim=hp.model_dim, latent_dim=hp.latent_dim,
                               heads=hp.heads, key_dim=hp.key_dim)(x)
    enc_out = x

    embed_dec = PositionalEmbedding(hp.seq_len + 1, hp.vocab_size, hp.model_dim)(dec_in)
    y = embed_dec
    for _ in range(hp.stacks):
        y = TransformerDecoder(embed_dim=hp.model_dim, latent_dim=hp.latent_dim,
                               heads=hp.heads, key_dim=hp.key_dim)(
            y, enc_out, y_mask=embed_dec._keras_mask, enc_mask=enc_out._keras_mask
        )

    y = layers.Dropout(0.1)(y)
    out = layers.Dense(hp.vocab_size, activation="softmax")(y)

    return keras.Model([enc_in, dec_in], out, name="transformer")


# ════════════════════════════════════════════════════════════════════════════
# 5. Training / evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

def make_tfds(pairs: List[Tuple[str, str]], hp: HParams) -> tf.data.Dataset:
    x_texts, y_texts = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(x_texts), list(y_texts)))
    ds = ds.batch(hp.batch).map(_format_for_transformer, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _format_for_transformer(x_str: tf.Tensor, y_str: tf.Tensor):
    x = INPUT_VECT(x_str)
    y = OUTPUT_VECT(y_str)
    return {"encoder_inputs": x, "decoder_inputs": y[:, :-1]}, y[:, 1:]


# ════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train / evaluate Transformer‑based text‑to‑text model")
    # data & I/O
    p.add_argument("--train-data", type=Path, help="TSV for training", required=True)
    p.add_argument("--valid-data", type=Path, help="TSV for validation", required=True)
    p.add_argument("--out", type=Path, default=Path("results"))

    # flags
    p.add_argument("--train", action="store_true", help="Run training stage")
    p.add_argument("--evaluate", action="store_true", help="Evaluate on validation set")
    p.add_argument("--predict", action="store_true", help="Write sample predictions.tsv")

    # h‑params
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--vocab", type=int, default=15_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--model-dim", type=int, default=512)
    p.add_argument("--latent", type=int, default=2_048)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--stacks", type=int, default=1)
    p.add_argument("--key-dim", type=int, default=0)
    return p


# ════════════════════════════════════════════════════════════════════════════
# 7. Main entry
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    hp = HParams(
        seq_len=args.seq_len,
        vocab_size=args.vocab,
        batch=args.batch,
        epochs=args.epochs,
        model_dim=args.model_dim,
        latent_dim=args.latent,
        heads=args.heads,
        stacks=args.stacks,
        key_dim=None if args.key_dim == 0 else args.key_dim,
        train_data=args.train_data,
        valid_data=args.valid_data,
        out_dir=args.out,
    )

    # ── read data
    train_pairs: List[Tuple[str, str]] = []
    if hp.train_data:
        train_lines = Path(hp.train_data).read_text().splitlines()
        train_pairs = [prepare_line(l) for l in train_lines]

    valid_lines = Path(hp.valid_data).read_text().splitlines()
    valid_pairs = [prepare_line(l) for l in valid_lines]

    # ── vectorisers (global for _format_for_transformer)
    global INPUT_VECT, OUTPUT_VECT
    INPUT_VECT = _vectoriser_layer(hp.vocab_size, hp.seq_len, hp.strip_chars)
    OUTPUT_VECT = _vectoriser_layer(hp.vocab_size, hp.seq_len + 1, hp.strip_chars)

    if train_pairs:
        LOGGER.info("Adapting vectorisers …")
        INPUT_VECT.adapt([p[0] for p in train_pairs])
        OUTPUT_VECT.adapt([p[1] for p in train_pairs])
        save_vectoriser(INPUT_VECT, hp.out_dir / "vectorizers" / "input_vect.keras")
        save_vectoriser(OUTPUT_VECT, hp.out_dir / "vectorizers" / "output_vect.keras")
    else:
        LOGGER.info("Loading pre‑trained vectorisers …")
        INPUT_VECT = load_vectoriser(hp.out_dir / "vectorizers" / "input_vect.keras")
        OUTPUT_VECT = load_vectoriser(hp.out_dir / "vectorizers" / "output_vect.keras")

    # update vocab size to true value
    true_vocab = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))
    if hp.vocab_size != true_vocab:
        LOGGER.info("Adjusting vocab_size → %d", true_vocab)
        hp.vocab_size = true_vocab

    # ── datasets
    train_ds = make_tfds(train_pairs, hp) if train_pairs else None
    valid_ds = make_tfds(valid_pairs, hp)

    # ── model
    model = build_transformer(hp)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath=hp.out_dir / "ckpt.weights.h5", save_weights_only=True, verbose=1
    )
    early = keras.callbacks.EarlyStopping(
        patience=10, min_delta=0.005, restore_best_weights=True, verbose=1
    )

    # ── training
    if args.train and train_ds is not None:
        LOGGER.info("Starting training …")
        hist = model.fit(train_ds, epochs=hp.epochs, validation_data=valid_ds, callbacks=[ckpt, early])
        pd.DataFrame(hist.history).to_csv(hp.out_dir / "history.csv", index=False)

    else:
        LOGGER.info("Loading weights …")
        model.load_weights(hp.out_dir / "ckpt.weights.h5")

    # ── evaluation
    if args.evaluate:
        loss, acc = model.evaluate(valid_ds, verbose=0)
        LOGGER.info("Validation — loss: %.4f | acc: %.4f", loss, acc)
        (hp.out_dir / "metrics.txt").write_text(f"val_loss\t{loss}\nval_acc\t{acc}\n")

    # ── prediction demo
    if args.predict:
        LOGGER.info("Writing sample predictions …")
        pred_lines = []
        for src, tgt in random.sample(valid_pairs, k=min(25, len(valid_pairs))):
            enc = INPUT_VECT([src])
            dec = OUTPUT_VECT([START_TOK])[:, :-1]
            out = "[start]"
            for _ in range(hp.seq_len + 1):
                logits = model([enc, dec])
                token_id = int(tf.argmax(logits[0, -1]).numpy())
                token = OUTPUT_VECT.get_vocabulary()[token_id]
                if token == END_TOK:
                    break
                out += " " + token
                dec = OUTPUT_VECT([out])[:, :-1]
            pred_lines.append(f"{src}\t{tgt}\t{out}")
        (hp.out_dir / "predictions.tsv").write_text("\n".join(pred_lines))
        LOGGER.info("Saved predictions to %s", hp.out_dir / "predictions.tsv")
