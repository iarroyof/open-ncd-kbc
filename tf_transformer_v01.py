"""
tf_transformer_v01_debug.py  – round 2
======================================

• Adds *no* model-architecture changes – only richer diagnostics.  
• Runs eagerly by default (no `@tf.function` on `tf_translate`).  
• Prints decoder length every generation step so we’ll see when it exceeds
  the positional-embedding table (`seq_len + 1`).  
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────
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
import os

# ── third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import mixed_precision
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

# ── logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("tf_transformer")

# ── hyper-parameters dataclass ────────────────────────────────────────────
@dataclass(kw_only=True)
class HParams:
    seq_len: int = 30
    dec_max_mult: int = 1 # 4
    vocab_size: int = 15_000
    model_dim: int = 512
    latent_dim: int = 2048
    heads: int = 1  # 8
    stacks: int = 1
    key_dim: int | None = None
    dropout: float = 0.3 # 0.1
    batch: int = 64
    epochs: int = 30
    train_path: str | None = None
    valid_path: str | None = None
    out_path: str = "results"
    seed: int = 42
    attn_sample_indices: List[int] | None = None
    debug: bool = False  # additional flag

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

# ── data utilities ────────────────────────────────────────────────────────
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
    src = f"{cols[2]} {pred}"
    tgt = f"{START} {cols[3]} {END}"
    return src, tgt

def parse_src(line: str) -> str:
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 4:
        raise ValueError("Each row needs ≥4 tab-separated fields for prediction")
    cols.pop(0)
    raw_pred = cols[1]
    pred = " ".join(re.findall(r"[A-Z][a-z]*", raw_pred)).lower() or raw_pred
    return f"{cols[2]} {pred}"

# ── vectorizer helpers ─────────────────────────────────────────────────────
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
    with zipfile.ZipFile(path, "r") as zf:
        zf.testzip()
    LOGGER.info("Saved and verified %s", path)

# ── attention logger ────────────────────────────────────────────────────
class AttentionLogger(keras.callbacks.Callback):
    """
    After every epoch draw attention heat-maps for a handful of
    validation sentences.  Only *decoder cross-attention* is plotted –
    that is, how each generated token attends to the encoded source.

    One PNG per block×head is written to <run_dir>/attn_ep_<N>.png and
    also logged as a W&B artefact.
    """
    def __init__(self, translator: Translator,
                 src_texts: list[str],
                 run_dir:   Path,
                 every_n_epochs: int = 1):
        super().__init__()
        self.t        = translator
        self.src      = src_texts
        self.run_dir  = run_dir
        self.every_n  = every_n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every_n:   # skip
            return

        # run a forward pass that captures cross-attn scores
        outs = self.t.tf_translate(tf.constant(self.src))
        preds = outs["text"].numpy().astype(str)

        # walk over decoder blocks
        for b_i, block in enumerate(self.t.model.layers):
            if not isinstance(block, DecBlock):
                continue
            scores = block.cross_scores[0]   # [heads, tgt, src]
            for h_i, head in enumerate(scores):
                plt.figure(figsize=(8, 4))
                sns.heatmap(head.numpy(),
                            xticklabels=self.src[0].split(),   # source tokens
                            yticklabels=preds[0].split(),      # predicted tokens
                            cmap="viridis")
                plt.xlabel("source")
                plt.ylabel("prediction")
                plt.title(f"epoch {epoch} – block {b_i} head {h_i}")
                fname = self.run_dir / f"attn_ep{epoch}_b{b_i}_h{h_i}.png"
                plt.tight_layout();  plt.savefig(fname); plt.close()
                wandb.log({fname.name: wandb.Image(str(fname))}, step=epoch)

# ── transformer layers ────────────────────────────────────────────────────
class MyLayerNorm(layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = self.add_weight("gamma", shape=(dim,), initializer="ones")
        self.beta  = self.add_weight("beta",  shape=(dim,), initializer="zeros")

    def call(self, x: tf.Tensor) -> tf.Tensor:                     # type: ignore[override]
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) * tf.math.rsqrt(var + self.eps) * self.gamma + self.beta

class PosEmbed(layers.Layer):
    def __init__(self, max_len: int, vocab: int, dim: int, name: str | None = None):
        super().__init__(name=name)
        self.max_len = max_len
        self.tok = layers.Embedding(vocab, dim, name="tok_emb")
        self.pos = layers.Embedding(max_len, dim, name="pos_emb")
        self.idx = tf.range(max_len)

    def call(self, x: tf.Tensor):                     # type: ignore[override]
        length = tf.shape(x)[-1]
        tf.print("[PosEmbed] actual_len =", length,
                 "/ max_len =", self.max_len,
                 ", input_shape =", tf.shape(x), summarize=-1)
        tf.debugging.assert_less_equal(
            length, self.max_len,
            message=(
                "Sequence length exceeds the positional-embedding table "
                f"of size {self.max_len}. "
                "Increase --dec-max-mult or stop the loop earlier."
            )
        )
        return self.tok(x) + self.pos(self.idx[:length])

    def compute_mask(self, *_) -> None: return None

class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout,
                                             name="enc_self",
                                             output_shape=None)
        self.mha_drop = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1, self.norm2 = MyLayerNorm(dim), MyLayerNorm(dim)

    def call(self, x: tf.Tensor, training=False):                  # type: ignore[override]
        attn_out, self.last_scores = self.mha(                     # <── NEW
            x, x, return_attention_scores=True,
            training=training)
        x = self.norm1(x + self.mha_drop(attn_out, training))
        return self.norm2(x + self.ffn_drop(self.ffn(x, training=training), training))

class DecBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.self_mha  = layers.MultiHeadAttention(heads, key_dim, dropout=dropout,
                                                   name="dec_self")
        self.cross_mha = layers.MultiHeadAttention(heads, key_dim, dropout=dropout,
                                                   name="dec_cross")
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.ffn_drop = layers.Dropout(dropout)
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)
        self.norm3 = MyLayerNorm(dim)

    def call(self, y: tf.Tensor, enc_out: tf.Tensor, training=False):  # type: ignore[override]
        sa_out,  self.self_scores  = self.self_mha(
            y, y, return_attention_scores=True, training=training)
        y = self.norm1(y + self.drop1(sa_out,  training))

        ca_out,  self.cross_scores = self.cross_mha(
            y, enc_out, return_attention_scores=True, training=training)
        y = self.norm2(y + self.drop2(ca_out, training))
        return self.norm3(y + self.ffn_drop(self.ffn(y, training=training), training))

# model builder
def build_model(h: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")

    x = PosEmbed(h.seq_len,     h.vocab_size, h.model_dim, name="pos_enc")(enc_in)
    for _ in range(h.stacks):
        x = EncBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, h.dropout)(x)
    enc_out = x

    dec_max_len = h.seq_len * h.dec_max_mult
    y = PosEmbed(dec_max_len, h.vocab_size, h.model_dim, name="pos_dec")(dec_in)
    for _ in range(h.stacks):
        y = DecBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim, h.dropout)(y, enc_out)
      
    y = layers.Dropout(h.dropout)(y)
    out = layers.Dense(h.vocab_size, activation="softmax", name="logits")(y)
    return keras.Model([enc_in, dec_in], out, name="transformer")

# masked loss + accuracy
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self): 
        super().__init__(reduction="none", name="masked_loss")
        self.base = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction="none"
        )

    def call(self, y_true, y_pred): 
        loss = self.base(y_true, y_pred)
        mask = tf.cast(y_true != 0, loss.dtype)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    mask = tf.cast(y_true != 0, tf.float32)
    preds = tf.argmax(y_pred, -1, output_type=tf.int64)
    match = tf.cast(tf.equal(tf.cast(y_true, tf.int64), preds), tf.float32)
    return tf.reduce_sum(match * mask) / tf.reduce_sum(mask)

# ── translator (eager) ────────────────────────────────────────────────────
class Translator(tf.Module):
    def __init__(
            self,
            model,
            in_tv,
            out_tv,
            *,
            temperature: float = 0.8,
            top_k: int = 40,
            top_p: float = 0.9):
        super().__init__()
        self.model = model
        self.in_tv, self.out_tv = in_tv, out_tv
        self.out_str = tf.keras.layers.StringLookup(
            vocabulary=out_tv.get_vocabulary(), mask_token="", invert=True
        )
        idx_from_str = tf.keras.layers.StringLookup(
            vocabulary=out_tv.get_vocabulary(), mask_token=""
        )
        self.start = idx_from_str("[start]")
        self.end   = idx_from_str("[end]")
        ban_ids = idx_from_str(["", "[UNK]", "[start]"]).numpy()
        self.token_mask = np.zeros(idx_from_str.vocabulary_size(), dtype=bool)
        self.token_mask[ban_ids] = True
        self.dec_max = model.get_layer("pos_dec").max_len
        # sampling hyper-params
        self.temperature = temperature
        self.top_k       = top_k
        self.top_p       = top_p              

    def tokens_to_text(self, tokens):
        return tf.strings.strip(
             tf.strings.reduce_join(
                 self.out_str(tokens), axis=1, separator=" "
             )
        )
    # ------------------------------------------------------------------
    #  Sampling helper – supports temperature, top-k and nucleus (top-p)
    # ------------------------------------------------------------------
    # ──────────────────────────────────────────────────────────────────────
    # Translator.sample  – temperature / top-k / top-p aware
    # ──────────────────────────────────────────────────────────────────────
    def sample(
        self,
        logits: tf.Tensor,
        *,                         # force keyword args for clarity
        temperature: float | None = None,
        top_k:       int   | None = None,
        top_p:       float | None = None,
    ) -> tf.Tensor:
        """
        Return a tensor **[batch]** with one token-id per batch element,
        applying (optional) temperature, top-k and/or nucleus (top-p) filtering.
    
        • If every control is “off’’ (τ==0 or None, k==0/None, p==0/None)
          the function degenerates to plain greedy decoding (arg-max).
    
        • Tokens whose ids are marked in `self.token_mask` are never sampled.
        """
        # ---- 1.  dtype hygiene --------------------------------------------------
        logits = tf.cast(logits, tf.float32)
    
        # ---- 2.  permanently ban unwanted ids ----------------------------------
        banned = self.token_mask[None, :]                    # [1,V]
        logits = tf.where(banned, tf.constant(-np.inf, tf.float32), logits)
    
        # ---- 3.  temperature ----------------------------------------------------
        τ = 0.0 if temperature is None else temperature
        if τ > 0.0:
            logits = logits / τ
    
        # ---- 4.  top-k ----------------------------------------------------------
        if top_k and top_k > 0:
            kth = tf.math.top_k(logits, k=top_k).values[:, -1, tf.newaxis]
            logits = tf.where(logits < kth, tf.constant(-np.inf, tf.float32), logits)
    
        # ---- 5.  top-p (nucleus) -----------------------------------------------
        if top_p and top_p > 0.0:
            sorted_logits = tf.sort(logits, axis=-1, direction="DESCENDING")
            cdf = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
            # index of *first* token where cumulative prob > p
            # ――― keep cut-off index in an *explicit* dtype (int32) ――――
            cut_idx = tf.argmax(
                tf.cast(cdf > top_p, tf.int32), axis=-1, output_type=tf.int32
            )
            # logit threshold corresponding to that index
            batch_idx = tf.range(tf.shape(logits)[0], dtype=tf.int32)  # <- same dtype
            thresh = tf.gather_nd(
                sorted_logits,
                tf.stack([batch_idx, cut_idx], axis=1),
            )
            logits = tf.where(logits < thresh[:, tf.newaxis],
                              tf.constant(-np.inf, tf.float32), logits)
    
        # ---- 6.  final choice ---------------------------------------------------
        greedy = (τ == 0.0) and (not top_k or top_k == 0) and (not top_p or top_p == 0.0)
        if greedy:
            return tf.argmax(logits, axis=-1, output_type=tf.int64)
    
        # categorical returns shape [B,1]  →  squeeze to [B]
        return tf.random.categorical(logits, 1, dtype=tf.int64)[:, 0]


    def translate(self, text, max_len=50):
        batch = tf.shape(text)[0]
        enc_in = self.in_tv(text)
        # (batch, 1) – keep rank-2 from the very beginning
        # keep the decoder sequence 2-D, but the “done” flag 1-D
        dec_in = tf.fill([batch, 1], self.start, name="dec_start")
        done   = tf.zeros([batch],      tf.bool)        # <-- rank-1
        out_tokens = []

        tf.print("[Translator] enc_inputs shape:", tf.shape(enc_in))
        for step in range(max_len):
            # SAFER: explicitly take the sequence dimension
            seq_len = tf.shape(dec_in)[1]
            if seq_len >= self.dec_max:
                tf.print("[Translator] reached dec_max_len → stopping early")
                break
            logits = self.model([enc_in, dec_in], training=False)[:, -1, :]
            # expand dims so rank == 2 → avoid accidental flattening
            new_tok = self.sample(
                logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            done |= (new_tok == self.end)               # shapes [B] ✓
            new_tok = tf.where(done, 0, new_tok)        # still [B] ✓

            new_tok = tf.expand_dims(new_tok, 1)        # [B,1]
            out_tokens.append(new_tok)
            dec_in = tf.concat([dec_in, new_tok], axis=1)  # ranks now match

            tf.print("[Translator] step", step, "dec_len", seq_len + 1)
            if tf.reduce_all(done): break

        return {"text": self.tokens_to_text(tf.concat(out_tokens, -1))}

    # --- NO @tf.function: runs eagerly so shape invariants are relaxed ---
    def tf_translate(self, inputs):
        return self.translate(inputs)

# ── dataset helpers ───────────────────────────────────────────────────────
INPUT_VECT: TextVectorization
OUTPUT_VECT: TextVectorization

def _fmt(src, tgt):
    src_tok = INPUT_VECT(src)
    tgt_tok = OUTPUT_VECT(tgt)
    return {"encoder_inputs": src_tok,
            "decoder_inputs": tgt_tok[:, :-1]}, tgt_tok[:, 1:]

def make_ds(pairs: List[Tuple[str, str]], h: HParams) -> tf.data.Dataset:
    s, t = zip(*pairs)
    return (tf.data.Dataset.from_tensor_slices((list(s), list(t)))
            .batch(h.batch).map(_fmt).prefetch(tf.data.AUTOTUNE))

# ── main ──────────────────────────────────────────────────────────────────
def main():
    global INPUT_VECT, OUTPUT_VECT

    # always run functions eagerly for debugging
    tf.config.run_functions_eagerly(True)

    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--train-path")
    p.add_argument("--valid-path", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out-path", default="results", help="root dir for outputs")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8,
                  help="0 = greedy decoding; >0 adds randomness")
    p.add_argument("--top-k",       type=int,   default=40,
                  help="keep only the K highest-logit tokens (0 = off)")
    p.add_argument("--top-p",       type=float, default=0.9,
                  help="nucleus sampling cumulative probability cutoff (0 = off)")
    p.add_argument("--dec-max-mult", type=int,   default=1,
                   help="decoder max-length multiplier vs. --seq-len")
    p.add_argument("--latent-dim",   type=int,   default=2048,
                   help="feed-forward network inner dimension")
    p.add_argument("--heads",        type=int,   default=1,
                   help="number of self-attention heads")
    p.add_argument("--stacks",       type=int,   default=1,
                   help="number of encoder / decoder blocks")
    p.add_argument("--attn-samples", type=int,   default=1,
                   help="how many validation sentences to plot attention for")

    manual_flags = {
        "--train-path", "--valid-path", "--out-path",
        "--epochs", "--temperature", "--top-k", "--top-p",
        "--train", "--evaluate", "--debug"
    }
    for fld in HParams.__dataclass_fields__:          # iterate every field
        flag = f"--{fld.replace('_','-')}"
        if flag in manual_flags:
            continue                                  # already declared
        p.add_argument(flag, type=int if fld != "debug" else bool)
    args = p.parse_args()

    cfg: dict = {k: v for k, v in vars(args).items()
                 if v is not None and k in HParams.__dataclass_fields__}
    cfg.setdefault("debug", args.debug)
    h = HParams(**cfg)

    if h.debug:
        LOGGER.setLevel(logging.DEBUG)

    mixed_precision.set_global_policy("mixed_float16")
    LOGGER.info("Mixed precision policy set → mixed_float16")

    train_pairs: list[Tuple[str, str]] = []
    if args.train:
        train_pairs = [parse_line(l)
                       for l in Path(args.train_path).read_text().splitlines()]
    valid_pairs = [parse_line(l)
                   for l in Path(args.valid_path).read_text().splitlines()]

    INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
    OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
    INPUT_VECT.adapt([s for s, _ in train_pairs or valid_pairs])
    OUTPUT_VECT.adapt([t for _, t in train_pairs or valid_pairs])
    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()),
                       len(OUTPUT_VECT.get_vocabulary()))

    train_ds = make_ds(train_pairs, h) if train_pairs else None
    valid_ds = make_ds(valid_pairs, h)

    model = build_model(h)
    opt = mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam())
    model.compile(optimizer=opt, loss=MaskedLoss(), metrics=[masked_accuracy])
    model.build([(None, None), (None, None)])
    model.summary(print_fn=LOGGER.info)

    # ---------------------------------------------------------------
    # create the run *first* so we have its metadata
    # ---------------------------------------------------------------
    run = wandb.init(                           # same pattern you used before
            project="tf-transformer",
            config=vars(args),                  # CLI flags in the config
    )
    cfg = run.config                            # optional shorthand

    # ---------------------------------------------------------------
    #     Derive the output folder
    #     <out-path>/<project>/<sweep-id|solo>/<run-id>/
    # ---------------------------------------------------------------
    project_name = run.project or "wandb_project"
    sweep_id     = run.sweep_id or "solo"       # .sweep_id is always present
    run_id       = run.id                       # always present

    run_root = os.path.join(                    # keep a final "/" for ease
        os.path.normpath(h.out_path), project_name, sweep_id, run_id
    ) + os.sep
    os.makedirs(run_root, exist_ok=True)

    # everything below keeps using `run_root` instead of `run_dir`
    run_dir = Path(run_root)                    # convenience Path object
    # save vectorisers after run_dir exists
    save_tv(INPUT_VECT,  run_dir / "input.keras")
    save_tv(OUTPUT_VECT, run_dir / "output.keras")
    callbacks = [wandb.keras.WandbCallback(save_model=False)]

    if train_ds:
        # pick the first N validation samples for visualisation
        sample_src = [s for s, _ in valid_pairs[: cfg.attn_samples]]
        attn_cb = AttentionLogger(
            translator = Translator(model, INPUT_VECT, OUTPUT_VECT,
                                    temperature=cfg.temperature,
                                    top_k=cfg.top_k,
                                    top_p=cfg.top_p),
            src_texts  = sample_src,
            run_dir    = run_dir)

        model.fit(train_ds,
                  validation_data = valid_ds,
                  epochs          = h.epochs,
                  callbacks       = callbacks + [attn_cb])
    if args.evaluate:
        try:
            tf.print("\n[MAIN] Starting translation evaluation …")
            t = Translator(
                    model, INPUT_VECT, OUTPUT_VECT,
                    temperature=cfg.temperature,   # e.g. 0.8
                    top_k=cfg.top_k,               # e.g. 40
                    top_p=cfg.top_p,               # e.g. 0.9
            )
            src, tgt_text = zip(*valid_pairs)         # gold answers
            preds = t.tf_translate(tf.constant(list(src)))["text"].numpy()
            pd.DataFrame(
               {"source": src, "prediction": preds, "target": tgt_text}
            ).to_csv(run_dir / "predictions.csv", index=False)
            LOGGER.info("Saved predictions → %s", run_dir / "predictions.csv")
        except Exception as exc:
            tb = "".join(traceback.format_tb(exc.__traceback__))
            err = run_dir / "evaluation_error.log"
            err.write_text(f"{exc}\n{tb}")
            LOGGER.error("Evaluation failed – see %s", err)
            raise

    wandb.finish()

if __name__ == "__main__":
    main()
