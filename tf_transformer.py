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
import json
from typing import List, Tuple

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
    path = path.with_suffix('')
    path.parent.mkdir(parents=True, exist_ok=True)
    config = tv.get_config()
    with open(path / "config.json", 'w') as f:
        json.dump(config, f)
    vocab = tv.get_vocabulary()
    with open(path / "vocab.txt", 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")
    LOGGER.info(f"Saved config to {path}/config.json and vocab to {path}/vocab.txt")

def load_tv(path: Path, custom_objects=None) -> TextVectorization:
    if custom_objects is None:
        custom_objects = {"standardize": standardize}
    path = path.with_suffix('')
    with open(path / "config.json", 'r') as f:
        config = json.load(f)
    tv = TextVectorization.from_config(config)
    with open(path / "vocab.txt", 'r') as f:
        vocab = [line.strip() for line in f]
    tv.adapt(["*init*"])
    tv.set_vocabulary(vocab)
    LOGGER.info(f"Loaded TextVectorization from {path}/config.json and {path}/vocab.txt")
    return tv

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

def predict(model, src_seq, max_len, start_token, end_token):
    enc_input = tf.expand_dims(src_seq, 0)
    dec_input = tf.expand_dims([start_token], 0)
    output = []
    for _ in range(max_len):
        predictions = model([enc_input, dec_input], training=False)
        last_token = tf.argmax(predictions[:, -1, :], axis=-1).numpy()[0]
        output.append(last_token)
        if last_token == end_token:
            break
        dec_input = tf.concat([dec_input, tf.expand_dims([last_token], 0)], axis=1)
    return output

# ════════════════════════════════════════════════════════════════════════════
# 8. Main execution
# ════════════════════════════════════════════════════════════════════════════

def main():
    global INPUT_VECT, OUTPUT_VECT
    LOGGER.info("Starting tf_transformer.py")

    parser = argparse.ArgumentParser(description="Train/evaluate Transformer model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--valid-path", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-path", type=str, default=None, help="Directory to load model and vectorizers for evaluation")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if args.evaluate and args.eval_path is None:
        parser.error("--eval-path is required when --evaluate is specified")
    if args.train and args.train_path is None:
        parser.error("--train-path is required when --train is specified")

    if args.evaluate:
        eval_path = Path(args.eval_path)
        with open(eval_path / "config.yaml", 'r') as f:
            config = yaml.safe_load(f) or {}
        h = HParams(**config)
        
        INPUT_VECT = load_tv(eval_path / "vectorizers" / "input")
        OUTPUT_VECT = load_tv(eval_path / "vectorizers" / "output")

        model = build_model(h)
        model.load_weights(eval_path / "ckpt.weights.h5")

        valid_lines = Path(h.valid_path).read_text().splitlines()
        src_list = [parse_src(l) for l in valid_lines]
        src_vect = INPUT_VECT(src_list).numpy()

        start_token = OUTPUT_VECT([START])[0, 0].numpy()
        end_token = OUTPUT_VECT([END])[0, 0].numpy()

        predictions = []
        for src in src_vect:
            pred = predict(model, src, h.seq_len, start_token, end_token)
            predictions.append(pred)

        vocab = OUTPUT_VECT.get_vocabulary()
        pred_texts = [" ".join([vocab[token] for token in pred if token != end_token]) for pred in predictions]

        with open(eval_path / "predictions.txt", 'w') as f:
            for text in pred_texts:
                f.write(text + "\n")
        LOGGER.info(f"Saved predictions to {eval_path / 'predictions.txt'}")
        return

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None and k in HParams.__dataclass_fields__:
            config[k] = v

    h = HParams(**config)

    if args.train_path is None:
        parser.error("--train-path is required for training or validation")

    train_lines = Path(h.train_path).read_text().splitlines()
    valid_lines = Path(h.valid_path).read_text().splitlines()
    train_pairs = [parse_line(l) for l in train_lines]
    valid_pairs = [parse_line(l) for l in valid_lines]

    INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
    OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
    INPUT_VECT.adapt([s for s, _ in train_pairs])
    OUTPUT_VECT.adapt([t for _, t in train_pairs])

    wandb_config = asdict(h)
    wandb.init(project="tf-transformer", config=wandb_config, save_code=True)

    run_out_dir = Path(h.out_dir) / wandb.run.project / (wandb.run.sweep_id or "nosweep") / wandb.run.id
    run_out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = h.to_dict()
    config_dict['out_dir'] = str(run_out_dir)
    with open(run_out_dir / "config.yaml", 'w') as f:
        yaml.dump(config_dict, f)

    h.out_dir = str(run_out_dir)

    save_tv(INPUT_VECT, Path(h.out_dir) / "vectorizers" / "input")
    save_tv(OUTPUT_VECT, Path(h.out_dir) / "vectorizers" / "output")
    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

    train_ds = make_ds(train_pairs, h) if args.train else None
    valid_ds = make_ds(valid_pairs, h)

    model = build_model(h)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    model.build(input_shape=[
        (None, None),
        (None, None),
    ])

    model.summary()

    if args.train:
        callbacks = [
            keras.callbacks.ModelCheckpoint(Path(h.out_dir) / "ckpt.weights.h5", save_weights_only=True, verbose=1),
            keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
            wandb.keras.WandbCallback(save_model=False),
        ]
        hist = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=h.epochs,
            callbacks=callbacks,
        )
        pd.DataFrame(hist.history).to_csv(Path(h.out_dir) / "history.csv", index=False)

if __name__ == "__main__":
    main()
