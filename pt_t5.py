#!/usr/bin/env python3
"""
Fine‑tune **T5‑small** on subject–predicate–object (SPO) triples with maximum backward‑compatibility to older 🤗 Transformers versions (no `predict_with_generate`).

Key points
-----------
* Same TSV → (input, target) preprocessing as the original TF‑GRU pipeline.
* W&B tracking + custom “probability of over‑fit” metric.
* Trainer without `predict_with_generate`; we call `model.generate()` manually for validation / hold‑out predictions.
* Script should run even on very old 4.x releases (down to ~4.0).
"""

import os
import re
import math
import string
import argparse
import logging
from functools import partial

import torch
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

STRIP_CHARS = string.punctuation.replace("[", "").replace("]", "")

def prepare_data(line: str,
                 start_token: str = "[start] ",
                 end_token: str = " [end]",
                 pmid: bool = True,
                 include_labels: bool = False,
                 include_sent: bool = False,
                 all_start_end: bool = True):
    """Convert one TSV row to (input, target) pair."""
    cols = line.rstrip("\n").split("\t")
    if pmid:
        cols.pop(0)
    predicate = " ".join(re.findall(r"[A-Z][a-z]*", cols[1])).lower() or cols[1]
    if not re.match(r"^-?\d+(?:\.\d+)?$", cols[4].strip()):
        extras = []
        i = 4
        while i < len(cols) and not re.match(r"^-?\d+(?:\.\d+)?$", cols[i].strip()):
            extras.append(cols.pop(i))
        cols[3] = " ".join([cols[3]] + extras)
    sample = [cols[0], predicate, cols[2], f"{start_token}{cols[3]}{end_token}", float(cols[4])]
    if include_labels:
        tgt = tuple(sample[-2:])
    else:
        sample.pop(-1)
        tgt = sample[-1]
    if include_sent:
        inp = " ".join([sample[0], sample[2], sample[1]])
    else:
        sample.pop(0)
        inp = " ".join([sample[1], sample[0]])
        if all_start_end:
            inp = f"{start_token}{inp}{end_token}"
    return inp, tgt

class OverfitCallback(TrainerCallback):
    def __init__(self, total_epochs: int, a=6.0, b=4.0, c=-2.0):
        self.total_epochs = total_epochs
        self.a, self.b, self.c = a, b, c
        self.epoch = 0
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        train_loss = metrics.get("loss")
        val_loss   = metrics.get("eval_loss")
        if train_loss is None or val_loss is None:
            return
        rel_gap = (val_loss - train_loss) / max(train_loss, 1e-8)
        epoch_ratio = (self.epoch + 1) / self.total_epochs
        p_overfit = 1 / (1 + math.exp(-(self.a*rel_gap + self.b*epoch_ratio + self.c)))
        wandb.log({"epoch": self.epoch+1, "rel_gap": rel_gap, "epoch_ratio": epoch_ratio,
                   "p_overfit": p_overfit, "train_loss": train_loss, "eval_loss": val_loss})
        self.epoch += 1

def generate_text(model, tokenizer, texts, max_len, device):
    """Generate outputs for a list of input strings."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model.generate(**enc, max_length=max_len+10)
    return tokenizer.batch_decode(outs, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser("Fine‑tune T5‑small for SPO generation")
    ap.add_argument("--trainData", required=True)
    ap.add_argument("--testData",  required=True)
    ap.add_argument("--holdoutData", default="")
    ap.add_argument("--modelName", default="t5-small")
    ap.add_argument("--seqLen", type=int, default=50)
    ap.add_argument("--batchSize", type=int, default=32)
    ap.add_argument("--nEpochs", type=int, default=10)
    ap.add_argument("--resPath", default=os.getcwd())
    args = ap.parse_args()

    run = wandb.init(project="t5_spo_generation", config=vars(args))
    cfg = run.config
    out_dir = os.path.join(cfg.resPath, run.project, run.id)
    os.makedirs(out_dir, exist_ok=True)

    with open(cfg.trainData) as f: train_lines = f.readlines()
    with open(cfg.testData)  as f: val_lines   = f.readlines()
    prep = partial(prepare_data, all_start_end=True)
    train_pairs = [prep(l) for l in train_lines]
    val_pairs   = [prep(l) for l in val_lines]
    train_inp, train_tgt = zip(*train_pairs)
    val_inp,   val_tgt   = zip(*val_pairs)

    tokenizer = T5Tokenizer.from_pretrained(cfg.modelName)
    model     = T5ForConditionalGeneration.from_pretrained(cfg.modelName)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def tok(batch):
        enc = tokenizer(batch["input"], max_length=cfg.seqLen, padding="max_length", truncation=True)
        dec = tokenizer(batch["target"], max_length=cfg.seqLen+1, padding="max_length", truncation=True)
        batch["input_ids"]      = enc.input_ids
        batch["attention_mask"] = enc.attention_mask
        batch["labels"]         = dec.input_ids
        return batch

    ds_train = Dataset.from_dict({"input": train_inp, "target": train_tgt}).map(tok, batched=True, remove_columns=["input","target"])
    ds_val   = Dataset.from_dict({"input": val_inp,   "target": val_tgt}).map(tok,   batched=True, remove_columns=["input","target"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.nEpochs,
        per_device_train_batch_size=cfg.batchSize,
        per_device_eval_batch_size=cfg.batchSize,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=ds_train,
                      eval_dataset=ds_val,
                      tokenizer=tokenizer,
                      data_collator=collator,
                      callbacks=[OverfitCallback(cfg.nEpochs)])

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Validation predictions
    logging.info("Generating validation predictions…")
    val_preds = generate_text(model, tokenizer, val_inp, cfg.seqLen, device)
    pd.DataFrame({"Subj_Pred": val_inp, "Obj": val_preds, "Obj_true": val_tgt}).to_csv(
        os.path.join(out_dir, "predictions.tsv"), sep="\t", index=False)

    # Hold‑out predictions
    if cfg.holdoutData and os.path.exists(cfg.holdoutData):
        with open(cfg.holdoutData) as f: hold_lines = f.readlines()
        hold_pairs = [prep(l) for l in hold_lines]
        hold_inp, hold_tgt = zip(*hold_pairs) if hold_pairs else ([], [])
        if hold_inp:
            logging.info("Generating hold‑out predictions…")
            hold_preds = generate_text(model, tokenizer, hold_inp, cfg.seqLen, device)
            pd.DataFrame({"Subj_Pred": hold_inp, "Obj": hold_preds, "Obj_true": hold_tgt}).to_csv(
                os.path.join(out_dir, "test_predictions.tsv"), sep="\t", index=False)

    wandb.finish()

if __name__ == "__main__":
    main()
