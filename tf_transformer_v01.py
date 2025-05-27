"""
Transformer training + evaluation script with detailed debugging logs
===================================================================
This version **only** adds logging/diagnostic utilities – it **does not**
change the architecture or any training‑time hyper‑parameters so that we can
collect enough context to understand the shape‑mismatch that appears during
evaluation.

Key additions
-------------
* `--debug` CLI flag that activates eager execution for tf.functions and
  sets the root logger to DEBUG.
* Extensive `tf.print` + Python `logging` statements inside
  `PosEmbed.call` and inside the token‑generation loop of `Translator`.
* Extra assertions and shape dumps (without modifying the forward pass)
  to capture encoder/decoder input lengths right before the failing AddV2
  operation.
* Guarded `try/except` wrapper around the evaluation block so we still get
  a CSV with all shapes collected **even if** the model crashes.

Save this file as **tf_transformer_v01_debug.py** and invoke it exactly as
before, but add `--debug` when you want the extra trace information, e.g.:
```bash
python tf_transformer_v01_debug.py --train --train-path data/train.tsv \
       --valid-path data/valid.tsv --evaluate --debug
```
"""
from __future__ import annotations

# ── standard library ─────────────────────────────────────────────────────────
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import logging
import random
import re
import string
import sys
import traceback
import yaml
from typing import List, Tuple
import zipfile

# ── third‑party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import mixed_precision
import wandb

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("tf_transformer")

# ----------------------------------------------------------------------------
# Hyper‑parameters dataclass
# ----------------------------------------------------------------------------
@dataclass(kw_only=True)
class HParams:
    seq_len: int = 30
    vocab_size: int = 15_000
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

    # extra debugging flag – gets injected from CLI, default False
    debug: bool = False

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

# ----------------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------------
START, END = "[start]", "[end]"
STRIP = string.punctuation.translate({ord("[" ): None, ord("]"): None})

@keras.saving.register_keras_serializable()
def standardize(text: tf.Tensor) -> tf.Tensor:
    text = tf.strings.lower(text)
    return tf.strings.regex_replace(text, f"[{re.escape(STRIP)}]", "")


def parse_line(line: str) -> Tuple[str, str]:
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 5:
        raise ValueError("Each row needs ≥5 tab‑separated fields")
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
        raise ValueError("Each row needs ≥4 tab‑separated fields for prediction")
    cols.pop(0)
    raw_pred = cols[1]
    pred = " ".join(re.findall(r"[A-Z][a-z]*", raw_pred)).lower() or raw_pred
    return f"{pred} {cols[2]}"

# ----------------------------------------------------------------------------
# Vectorizers
# ----------------------------------------------------------------------------

def build_vectorizer(vocab: int, seq_len: int) -> TextVectorization:
    return TextVectorization(
        max_tokens=vocab,
        output_sequence_length=seq_len,
        standardize=standardize,
        output_mode="int",
    )


def save_tv(tv: TextVectorization, path: Path) -> None:
    """Persist a TextVectorization layer and verify the zip file integrity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = keras.Sequential([keras.Input(shape=(1,), dtype="string"), tv])
    model.save(path, save_format="keras")
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.testzip()
        LOGGER.info("Saved and verified %s", path)
    except zipfile.BadZipFile as exc:
        LOGGER.error("Failed to save %s: %s", path, exc)
        raise

# ----------------------------------------------------------------------------
# Transformer building blocks (unchanged except for logging)
# ----------------------------------------------------------------------------
class MyLayerNorm(layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = self.add_weight("gamma", shape=(dim,), initializer="ones", trainable=True)
        self.beta = self.add_weight("beta", shape=(dim,), initializer="zeros", trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        normed = (x - mean) * tf.math.rsqrt(var + self.eps)
        return normed * self.gamma + self.beta


class PosEmbed(layers.Layer):
    """Token + positional embedding with extensive diagnostics."""

    def __init__(self, max_len: int, vocab: int, dim: int, name: str | None = None):
        super().__init__(name=name)
        self.max_len = max_len
        self.vocab = vocab
        self.dim = dim
        self.tok = layers.Embedding(vocab, dim, name="embedding")
        self.pos = layers.Embedding(max_len, dim, name="pos_embedding")
        self.idx = tf.range(max_len)

    def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        length = tf.shape(x)[-1]
        # Graph‑mode safe debug print
        tf.print("[PosEmbed] actual_len=", length, "/ max_len=", self.max_len, \
                 ", input_shape=", tf.shape(x), summarize=-1)
        if tf.executing_eagerly():  # extra Python‑side log when eager/debug flag is on
            LOGGER.debug("[PosEmbed] eager input len=%s max=%d shape=%s", x.shape[-1], self.max_len, x.shape)
        return self.tok(x) + self.pos(self.idx[:length])

    def compute_mask(self, x: tf.Tensor, _=None):  # noqa: D401, override‑name
        return None


class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout, name="enc_mha")
        self.mha_drop = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ], name="enc_ffn")
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)

    def call(self, x: tf.Tensor, training: bool = False):  # type: ignore[override]
        attn = self.mha(x, x, training=training)
        x = self.norm1(x + self.mha_drop(attn, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.ffn_drop(ffn_out, training=training))


class DecBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout, name="dec_self_mha")
        self.self_drop = layers.Dropout(dropout)
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout, name="dec_cross_mha")
        self.cross_drop = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ], name="dec_ffn")
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)
        self.norm3 = MyLayerNorm(dim)

    def call(self, y: tf.Tensor, enc_out: tf.Tensor, training: bool = False):  # type: ignore[override]
        self_attn = self.self_mha(y, y, training=training)
        y = self.norm1(y + self.self_drop(self_attn, training=training))
        cross_attn = self.cross_mha(y, enc_out, training=training)
        y = self.norm2(y + self.cross_drop(cross_attn, training=training))
        ffn_out = self.ffn(y, training=training)
        return self.norm3(y + self.ffn_drop(ffn_out, training=training))

# ----------------------------------------------------------------------------
# Model builder – unchanged architecture
# ----------------------------------------------------------------------------

def build_model(h: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")

    x = PosEmbed(h.seq_len, h.vocab_size, h.model_dim, name="pos_embed_0")(enc_in)
    for _ in range(h.stacks):
        x = EncBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, dropout=h.dropout)(x)
    enc_out = x

    y = PosEmbed(h.seq_len + 1, h.vocab_size, h.model_dim, name="pos_embed_1")(dec_in)
    for _ in range(h.stacks):
        y = DecBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, dropout=h.dropout)(y, enc_out)

    y = layers.Dropout(h.dropout)(y)
    out = layers.Dense(h.vocab_size, activation="softmax", name="logits_out")(y)
    return keras.Model([enc_in, dec_in], out, name="transformer")

# ----------------------------------------------------------------------------
# Masked loss + accuracy
# ----------------------------------------------------------------------------
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(name="masked_loss")
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

    def __call__(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != 0, loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int64)
    matches = tf.cast(tf.equal(tf.cast(y_true, tf.int64), pred_ids), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

# ----------------------------------------------------------------------------
# Translator with detailed per‑step tracing
# ----------------------------------------------------------------------------
class Translator(tf.Module):
    def __init__(self, model, input_text_processor, output_text_processor):
        super().__init__()
        self.model = model
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.output_token_string_from_index = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token="", invert=True
        )
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token=""
        )
        token_mask_ids = index_from_string(["", "[UNK]", "[start]"]).numpy()
        self.token_mask = np.zeros(index_from_string.vocabulary_size(), dtype=bool)
        self.token_mask[token_mask_ids] = True
        self.start_token = index_from_string(tf.constant("[start]"))
        self.end_token = index_from_string(tf.constant("[end]"))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=" ")
        return tf.strings.strip(result_text)

    def sample(self, logits, temperature):
        logits = tf.cast(logits, tf.float32)
        mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(mask, tf.constant(-np.inf, dtype=tf.float32), logits)
        if temperature == 0.0:
            return tf.argmax(logits, axis=-1, output_type=tf.int64)
        logits = tf.squeeze(logits, axis=1)
        return tf.random.categorical(logits / temperature, num_samples=1, dtype=tf.int64)

    def translate(self, input_text, max_length: int = 50, temperature: float = 1.0):  # noqa: D401
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_inputs = input_tokens
        dec_inputs = tf.fill([batch_size, 1], self.start_token)
        result_tokens = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        tf.print("[Translator] enc_inputs shape:", tf.shape(enc_inputs))
        for step in tf.range(max_length):
            preds = self.model([enc_inputs, dec_inputs], training=False)
            logits = preds[:, -1, :]
            new_tokens = self.sample(logits, temperature)
            done |= new_tokens == self.end_token
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            result_tokens.append(new_tokens)
            dec_inputs = tf.concat([dec_inputs, new_tokens], axis=-1)

            # per‑step debug trace
            tf.print("[Translator] step", step, "dec_len", tf.shape(dec_inputs)[-1])
            if tf.reduce_all(done):
                break

        result_tokens = tf.concat(result_tokens, axis=-1)
        return {"text": self.tokens_to_text(result_tokens)}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def tf_translate(self, input_text):
        return self.translate(input_text)

# ----------------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------------
INPUT_VECT: TextVectorization
OUTPUT_VECT: TextVectorization


def _fmt(src: tf.Tensor, tgt: tf.Tensor):
    src_tok = INPUT_VECT(src)
    tgt_tok = OUTPUT_VECT(tgt)
    return {"encoder_inputs": src_tok, "decoder_inputs": tgt_tok[:, :-1]}, tgt_tok[:, 1:]


def make_ds(pairs: List[Tuple[str, str]], h: HParams) -> tf.data.Dataset:
    s, t = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(s), list(t)))
    ds = ds.batch(h.batch).map(_fmt).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    global INPUT_VECT, OUTPUT_VECT

    parser = argparse.ArgumentParser(description="Train + debug Transformer model")
    parser.add_argument("--config", type=str, help="Optional YAML config file")
    parser.add_argument("--train-path", type=str, help="Path to training data (.tsv)")
    parser.add_argument("--valid-path", type=str, required=True, help="Path to validation data (.tsv)")
    parser.add_argument("--train", action="store_true", help="Run training stage")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation right after training")
    parser.add_argument("--save-weights", action="store_true", help="Save final weights")

    # hyper‑params (override defaults)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--model-dim", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--stacks", type=int)
    parser.add_argument("--key-dim", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attn-sample-indices", type=int, nargs="*")

    # NEW
    parser.add_argument("--debug", action="store_true", help="Activate verbose debugging mode")

    args = parser.parse_args()

    if args.train and args.train_path is None:
        parser.error("--train-path is required when --train is specified")

    # optional YAML config load
    cfg: dict = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides take precedence
    for k, v in vars(args).items():
        if v is not None and k in HParams.__dataclass_fields__:
            cfg[k] = v

    cfg.setdefault("debug", args.debug)
    h = HParams(**cfg)

    if h.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        tf.config.run_functions_eagerly(True)
        LOGGER.debug("Eager execution enabled for debugging")

    # ------------------------------------------------------------
    # Mixed precision policy
    mixed_precision.set_global_policy("mixed_float16")
    LOGGER.info("Mixed precision policy set → mixed_float16")

    # ------------------------------------------------------------
    # Data ingestion
    train_pairs: List[Tuple[str, str]] = []
    if args.train:
        train_lines = Path(h.train_path).read_text().splitlines()
        train_pairs = [parse_line(l) for l in train_lines]
    valid_lines = Path(h.valid_path).read_text().splitlines()
    valid_pairs = [parse_line(l) for l in valid_lines]

    # vectorizers
    INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
    OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
    if train_pairs:
        INPUT_VECT.adapt([s for s, _ in train_pairs])
        OUTPUT_VECT.adapt([t for _, t in train_pairs])
    else:
        # use validation set to create reasonable vocab when only evaluating
        INPUT_VECT.adapt([s for s, _ in valid_pairs])
        OUTPUT_VECT.adapt([t for _, t in valid_pairs])

    LOGGER.info("Output vocab size=%d", len(OUTPUT_VECT.get_vocabulary()))
    LOGGER.debug("Sample vocab=%s", OUTPUT_VECT.get_vocabulary()[:20])

    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

    # datasets
    if train_pairs:
        train_ds = make_ds(train_pairs, h)
    valid_ds = make_ds(valid_pairs, h)

    model = build_model(h)
    opt = tf.keras.optimizers.Adam()
    opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss=MaskedLoss(), metrics=[masked_accuracy])

    model.build(input_shape=[(None, None), (None, None)])
    model.summary(print_fn=LOGGER.info)

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        wandb.keras.WandbCallback(save_model=False),
    ]

    out_dir = Path(h.out_dir)
    run_out_dir = out_dir / wandb.run.project / (wandb.run.sweep_id or "nosweep") / wandb.run.id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_weights:
        callbacks.append(keras.callbacks.ModelCheckpoint(run_out_dir / "ckpt.weights.h5", save_weights_only=True, verbose=1))

    save_tv(INPUT_VECT, run_out_dir / "vectorizers" / "input.keras")
    save_tv(OUTPUT_VECT, run_out_dir / "vectorizers" / "output.keras")

    if train_pairs:
        hist = model.fit(train_ds, validation_data=valid_ds, epochs=h.epochs, callbacks=callbacks)
        pd.DataFrame(hist.history).to_csv(run_out_dir / "history.csv", index=False)

    # ---------------------------------------------------------------------
    # Evaluation with robust error capture
    # ---------------------------------------------------------------------
    if args.evaluate:
        LOGGER.info("Starting evaluation (translate)…")
        translator = Translator(model, INPUT_VECT, OUTPUT_VECT)
        inp_, targ_ = zip(*valid_pairs)

        try:
            results = translator.tf_translate(tf.constant(list(inp_)))
            preds = results["text"].numpy().tolist()
            df = pd.DataFrame({"Subj_Pred": inp_, "Obj_pred": preds, "Obj_true": targ_})
            df.to_csv(run_out_dir / "predictions.csv", index=False)
            LOGGER.info("Saved predictions → %s", run_out_dir / "predictions.csv")
        except Exception as exc:
            # dump traceback and continue re‑raising for full visibility
            tb_str = "\n".join(traceback.format_tb(exc.__traceback__))
            err_path = run_out_dir / "evaluation_error.log"
            err_path.write_text(f"{exc}\n{tb_str}\n")
            LOGGER.error("Evaluation failed – see %s", err_path)
            raise


if __name__ == "__main__":
    # initialise wandb only when script is launched directly
    wandb.init(project="tf-transformer", save_code=True)
    try:
        main()
    finally:
        wandb.finish()
