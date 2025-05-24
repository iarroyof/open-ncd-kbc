import argparse
import tensorflow as tf
import wandb
import yaml
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import keras

# Assuming these are defined elsewhere in the script
from some_module import HParams, TextVectorization, build_vectorizer, save_tv, load_tv, parse_line, parse_src, build_model, START, END

INPUT_VECT: TextVectorization
OUTPUT_VECT: TextVectorization

def _fmt(src: tf.Tensor, tgt: tf.Tensor):
    src_tok = INPUT_VECT(src)
    tgt_tok = OUTPUT_VECT(tgt)
    return {"encoder_inputs": src_tok, "decoder_inputs": tgt_tok[:, :-1]}, tgt_tok[:, 1:]

def make_ds(pairs: list[tuple[str, str]], h: HParams) -> tf.data.Dataset:
    s, t = zip(*pairs)
    ds = tf.data.Dataset.from_tensor_slices((list(s), list(t)))
    return ds.batch(h.batch).map(_fmt).prefetch(tf.data.AUTOTUNE)

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

def main():
    global INPUT_VECT, OUTPUT_VECT  # Declare global at the start

    parser = argparse.ArgumentParser(description="Train/evaluate Transformer model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-dir", type=str, default=None, help="Directory to load model and vectorizers for evaluation")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    if args.evaluate and args.eval_dir is None:
        parser.error("--eval-dir is required when --evaluate is specified")

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    args_dict = vars(args)
    for k, v in args_dict.items():
        if v is not None and k in HParams.__dataclass_fields__:
            config[k] = v

    h = HParams(**config)

    if not args.evaluate:
        train_lines = h.train_path.read_text().splitlines()
        valid_lines = h.valid_path.read_text().splitlines()
        train_pairs = [parse_line(l) for l in train_lines]
        valid_pairs = [parse_line(l) for l in valid_lines]

        INPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len)
        OUTPUT_VECT = build_vectorizer(h.vocab_size, h.seq_len + 1)
        INPUT_VECT.adapt([s for s, _ in train_pairs])
        OUTPUT_VECT.adapt([t for _, t in train_pairs])

        wandb_config = {k: v for k, v in asdict(h).items() if not isinstance(v, Path)}
        wandb.init(project="tf-transformer", config=wandb_config, save_code=True)

        run_out_dir = h.out_dir / wandb.run.project / (wandb.run.sweep_id or "nosweep") / wandb.run.id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        h.out_dir = run_out_dir

        save_tv(INPUT_VECT, h.out_dir / "vectorizers" / "input.keras")
        save_tv(OUTPUT_VECT, h.out_dir / "vectorizers" / "output.keras")
        h.vocab_size = max(len(INPUT_VECT.get_vocabulary()), len(OUTPUT_VECT.get_vocabulary()))

        with open(h.out_dir / "config.yaml", 'w') as f:
            yaml.dump(asdict(h), f)

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
                keras.callbacks.ModelCheckpoint(h.out_dir / "ckpt.weights.h5", save_weights_only=True, verbose=1),
                keras.callbacks.EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True, verbose=1),
                wandb.keras.WandbCallback(save_model=False),
            ]
            hist = model.fit(
                train_ds,
                validation_data=valid_ds,
                epochs=h.epochs,
                callbacks=callbacks,
            )
            pd.DataFrame(hist.history).to_csv(h.out_dir / "history.csv", index=False)

    if args.evaluate:
        eval_dir = Path(args.eval_dir)
        with open(eval_dir / "config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        h = HParams(**config)
        
        INPUT_VECT = load_tv(eval_dir / "vectorizers" / "input.keras")
        OUTPUT_VECT = load_tv(eval_dir / "vectorizers" / "output.keras")

        model = build_model(h)
        model.load_weights(eval_dir / "ckpt.weights.h5")

        valid_lines = h.valid_path.read_text().splitlines()
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

        with open(eval_dir / "predictions.txt", 'w') as f:
            for text in pred_texts:
                f.write(text + "\n")

if __name__ == "__main__":
    main()
