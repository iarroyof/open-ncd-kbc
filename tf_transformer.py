
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
from datetime import datetime

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import wandb
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import seaborn as sns
import matplotlib.pyplot as plt

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
    train_path: Path | None = None
    valid_path: Path | None = None
    out_dir: Path = Path("results")
    seed: int = 42

    def __post_init__(self):
        # Convert strings to Path objects
        if self.train_path is not None:
            self.train_path = Path(str(self.train_path))
        if self.valid_path is not None:
            self.valid_path = Path(str(self.valid_path))
        self.out_dir = Path(str(self.out_dir))

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
STRIP = string.punctuation.translate({ord("["): None, ord("]"): None})

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
    if not cols[4].isdigit() and not re.match(r"^-?\d+(?:\.\d+)?$", cols[4]):
        extras = []
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

def load_tv(path: Path, custom_objects=None) -> TextVectorization:
    mdl = keras.models.load_model(path, custom_objects=custom_objects)
    old: TextVectorization = mdl.layers[1]
    cfg, vocab = old.get_config(), old.get_vocabulary()
    new = TextVectorization.from_config(cfg)
    new.adapt(["_init_"])
    new.set_vocabulary(vocab)
    return new

# ════════════════════════════════════════════════════════════════════════════
# 4. Transformer building blocks
# ════════════════════════════════════════════════════════════════════════════
class MyLayerNorm(layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = self.add_weight('gamma', shape=(dim,), initializer='ones', trainable=True)
        self.beta = self.add_weight('beta', shape=(dim,), initializer='zeros', trainable=True)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        normed = (x - mean) * tf.math.rsqrt(var + self.eps)
        return normed * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "eps": self.eps
        })
        return config

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

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "vocab": self.vocab,
            "dim": self.dim
        })
        return config

class EncBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.heads = heads
        self.key_dim = key_dim
        self.mha = layers.MultiHeadAttention(heads, key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(latent, activation="relu"),
            layers.Dense(dim),
        ])
        self.norm1 = MyLayerNorm(dim)
        self.norm2 = MyLayerNorm(dim)

    def call(self, x: tf.Tensor, mask: tf.Tensor | None = None, training: bool = False) -> tf.Tensor:
        attn_output, attn_weights = self.mha(
            tf.cast(x, tf.float32), tf.cast(x, tf.float32), return_attention_scores=True, training=training
        )
        x = self.norm1(tf.cast(x, tf.float32) + tf.cast(attn_output, tf.float32))
        ffn_out = self.ffn(tf.cast(x, tf.float32), training=training)
        x = self.norm2(tf.cast(x, tf.float32) + tf.cast(ffn_out, tf.float32))
        return x, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "latent": self.latent,
            "heads": self.heads,
            "key_dim": self.key_dim
        })
        return config

class DecBlock(layers.Layer):
    def __init__(self, dim: int, latent: int, heads: int, key_dim: int):
        super().__init__()
        self.dim = dim
        self.latent = latent
        self.heads = heads
        self.key_dim = key_dim
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
        self_attn_output, self_attn_weights = self.self_mha(
            tf.cast(y, tf.float32), tf.cast(y, tf.float32), return_attention_scores=True, training=training
        )
        y = self.norm1(tf.cast(y, tf.float32) + tf.cast(self_attn_output, tf.float32))
        cross_attn_output, cross_attn_weights = self.cross_mha(
            tf.cast(y, tf.float32), tf.cast(enc_out, tf.float32), return_attention_scores=True, training=training
        )
        y = self.norm2(tf.cast(y, tf.float32) + tf.cast(cross_attn_output, tf.float32))
        ffn_out = self.ffn(tf.cast(y, tf.float32), training=training)
        y = self.norm3(tf.cast(y, tf.float32) + tf.cast(ffn_out, tf.float32))
        return y, self_attn_weights, cross_attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "latent": self.latent,
            "heads": self.heads,
            "key_dim": self.key_dim
        })
        return config

# ════════════════════════════════════════════════════════════════════════════
# 5. Model builder
# ════════════════════════════════════════════════════════════════════════════
def build_model(h: HParams) -> keras.Model:
    enc_in = keras.Input((None,), dtype="int64", name="encoder_inputs")
    dec_in = keras.Input((None,), dtype="int64", name="decoder_inputs")
    x = PosEmbed(h.seq_len, h.vocab_size, h.model_dim)(enc_in)
    enc_attn_weights = []
    for _ in range(h.stacks):
        x, attn_weights = EncBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim)(x)
        enc_attn_weights.append(attn_weights)
    enc_out = x
    y = PosEmbed(h.seq_len + 1, h.vocab_size, h.model_dim)(dec_in)
    self_attn_weights = []
    cross_attn_weights = []
    for _ in range(h.stacks):
        y, self_attn, cross_attn = DecBlock(h.model_dim, h.latent_dim, h.heads, h.key_dim)(y, enc_out)
        self_attn_weights.append(self_attn)
        cross_attn_weights.append(cross_attn)
    y = layers.Dropout(0.1)(y)
    out = layers.Dense(h.vocab_size, activation="softmax")(y)
    return keras.Model(
        [enc_in, dec_in],
        [out] + enc_attn_weights + self_attn_weights + cross_attn_weights,
        name="transformer"
    )

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
# 7. Helper functions for attention and metrics
# ════════════════════════════════════════════════════════════════════════════
def indices_to_tokens(indices: np.ndarray, vectorizer: TextVectorization) -> List[str]:
    """Convert token indices to tokens, excluding padding and special tokens."""
    vocab = vectorizer.get_vocabulary()
    tokens = [vocab[idx] for idx in indices if idx > 0 and vocab[idx] not in [START, END, '[PAD]']]
    return tokens

def log_attention_heatmap(attn_weights: np.ndarray, src_tokens: List[str], tgt_tokens: List[str], name: str, epoch: int):
    """Log attention matrix as a heatmap to W&B."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, xticklabels=tgt_tokens, yticklabels=src_tokens, cmap="viridis")
    plt.title(f"{name} Attention (Epoch {epoch})")
    wandb.log({f"attention/{name}_epoch_{epoch}": wandb.Image(plt)})
    plt.close()

def compute_metrics(pred_texts: List[str], ref_texts: List[str]) -> dict:
    """Compute ROUGE, BLEU, and METEOR scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    bleu_scores = []
    meteor_scores = []
    
    for pred, ref in zip(pred_texts, ref_texts):
        # ROUGE
        scores = scorer.score(ref, pred)
        for key in rouge_scores:
            rouge_scores[key] += scores[key].fmeasure
        # BLEU
        bleu_scores.append(sentence_bleu([ref.split()], pred.split()))
        # METEOR
        meteor_scores.append(meteor_score([ref.split()], pred.split()))
    
    n = len(pred_texts)
    return {
        'rouge1': rouge_scores['rouge1'] / n,
        'rouge2': rouge_scores['rouge2'] / n,
        'rougeL': rouge_scores['rougeL'] / n,
        'bleu': np.mean(bleu_scores),
        'meteor': np.mean(meteor_scores)
    }

def decode_sequence(model, src: np.ndarray, h: HParams, output_vectorizer: TextVectorization) -> str:
    """Decode a single source sequence to predicted target."""
    src = src[np.newaxis, :]  # Add batch dimension
    dec_input = output_vectorizer([[START]])[:, :-1]
    for _ in range(h.seq_len):
        predictions = model.predict([src, dec_input], verbose=0)
        next_token = np.argmax(predictions[0][0, -1])
        if output_vectorizer.get_vocabulary()[next_token] == END:
            break
        dec_input = np.concatenate([dec_input, [[next_token]]], axis=-1)
    tokens = indices_to_tokens(dec_input[0], output_vectorizer)
    return " ".join(tokens)

# ════════════════════════════════════════════════════════════════════════════
# 8. Main execution
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Transformer model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--project", default="tf-transformer")
    parser.add_argument("--sweep", default="default")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=5, help="Number of random validation samples to evaluate")
    parser.add_argument("--eval-file", type=str, default=None, help="Text file with source sentences for evaluation")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Override config with CLI args
    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None and k in HParams.__dataclass_fields__:
            config[k] = v

    h = HParams(**config)

    # Set up run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / args.project / args.sweep / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vectorizers").mkdir(exist_ok=True)  # Ensure vectorizers directory exists
    h.out_dir = run_dir

    # Load and parse data
    train_lines = Path(args.train_path).read_text().splitlines()
    valid_lines = Path(args.valid_path).read_text().splitlines()
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
    h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

    train_ds = make_ds(train_pairs, h) if args.train else None
    valid_ds = make_ds(valid_pairs, h)

    # Build and compile model
    model = build_model(h)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    # Explicitly build the model with input shapes
    model.build(input_shape=[
        (None, None),  # encoder_inputs shape: (batch_size, seq_len)
        (None, None),  # decoder_inputs shape: (batch_size, seq_len)
    ])
    model.summary()

    # WandB setup
    wandb_config = {k: v for k, v in asdict(h).items() if not isinstance(v, Path)}
    wandb_config.update({"run_dir": str(run_dir)})
    wandb.init(project=args.project, config=wandb_config, save_code=True)

    # Training
    if args.train:
        checkpoint_path = h.out_dir / "ckpt.weights.h5"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            LOGGER.info(f"Deleted existing checkpoint file: {checkpoint_path}")

        class AttentionAndMetricsCallback(keras.callbacks.Callback):
            def __init__(self, valid_pairs, input_vectorizer, output_vectorizer, h):
                super().__init__()
                self.valid_pairs = valid_pairs
                self.input_vectorizer = input_vectorizer
                self.output_vectorizer = output_vectorizer
                self.h = h

            def on_epoch_end(self, epoch, logs=None):
                # Sample one validation example for attention
                src, tgt = random.choice(self.valid_pairs)
                src_tok = self.input_vectorizer([[src]])
                dec_input = self.output_vectorizer([[START]])[:, :-1]
                outputs = self.model.predict([src_tok, dec_input], verbose=0)
                pred_probs = outputs[0]
                attn_weights = outputs[1:]  # Encoder, self-attention, cross-attention weights

                # Convert indices to tokens
                src_tokens = indices_to_tokens(src_tok[0].numpy(), self.input_vectorizer)
                pred_indices = np.argmax(pred_probs[0], axis=-1)
                pred_tokens = indices_to_tokens(pred_indices, self.output_vectorizer)

                # Log attention matrices
                for i, weights in enumerate(attn_weights):
                    if i < self.h.stacks:
                        log_attention_heatmap(weights[0, 0], src_tokens, src_tokens, f"enc_block_{i+1}", epoch + 1)
                    elif i < 2 * self.h.stacks:
                        log_attention_heatmap(weights[0, 0], pred_tokens, pred_tokens, f"dec_self_block_{i-self.h.stacks+1}", epoch + 1)
                    else:
                        log_attention_heatmap(weights[0, 0], src_tokens, pred_tokens, f"dec_cross_block_{i-2*self.h.stacks+1}", epoch + 1)

                # Compute metrics
                pred_texts = []
                ref_texts = [t.replace(START, "").replace(END, "").strip() for _, t in self.valid_pairs]
                for src, _ in self.valid_pairs:
                    pred = decode_sequence(self.model, self.input_vectorizer([[src]])[0], self.h, self.output_vectorizer)
                    pred_texts.append(pred)
                metrics = compute_metrics(pred_texts, ref_texts)
                wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=epoch + 1)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_weights_only=True,
                overwrite=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
            wandb.keras.WandbCallback(save_model=False),
            AttentionAndMetricsCallback(valid_pairs, INPUT_VECT, OUTPUT_VECT, h),
        ]
        LOGGER.info("Starting model.fit with checkpoint path: %s", checkpoint_path)
        hist = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=h.epochs,
            callbacks=callbacks,
        )
        LOGGER.info("Training completed successfully")
        pd.DataFrame(hist.history).to_csv(h.out_dir / "history.csv", index=False)

    # Evaluation
    if args.evaluate:
        # Load model and vectorizers
        custom_objects = {"MyLayerNorm": MyLayerNorm, "PosEmbed": PosEmbed, "EncBlock": EncBlock, "DecBlock": DecBlock}
        model.load_weights(h.out_dir / "ckpt.weights.h5")
        input_vectorizer = load_tv(h.out_dir / "vectorizers" / "input.keras", custom_objects=custom_objects)
        output_vectorizer = load_tv(h.out_dir / "vectorizers" / "output.keras", custom_objects=custom_objects)

        # Get source sentences
        if args.eval_file:
            with open(args.eval_file, 'r') as f:
                sources = [line.strip() for line in f if line.strip()]
            pairs = [(src, None) for src in sources]
        else:
            pairs = random.sample(valid_pairs, min(args.eval_samples, len(valid_pairs)))

        # Generate predictions
        print("src\tpred_trg\ttrg")
        for src, tgt in pairs:
            pred = decode_sequence(model, input_vectorizer([[src]])[0], h, output_vectorizer)
            trg = tgt.replace(START, "").replace(END, "").strip() if tgt else ""
            print(f"{src}\t{pred}\t{trg}")

        # Log metrics to W&B
        if not args.eval_file:  # Only compute metrics for validation data
            pred_texts = [decode_sequence(model, input_vectorizer([[src]])[0], h, output_vectorizer) for src, _ in valid_pairs]
            ref_texts = [t.replace(START, "").replace(END, "").strip() for _, t in valid_pairs]
            metrics = compute_metrics(pred_texts, ref_texts)
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})

if __name__ == "__main__":
    main()
