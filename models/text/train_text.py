"""Fine-tune a pretrained encoder for SST-2 and save it as a text checkpoint.

Run once per base model with the same settings, so the comparison is fair:

    python -m models.text.train_text --base bert-base-uncased --out models/text/bert_base_uncased.pt
    python -m models.text.train_text --base roberta-base      --out models/text/roberta_base.pt

Training uses data/sst2/train.jsonl only. The validation split is used only to
report accuracy after training. The .pt files are large and are not committed
(see .gitignore); rerun this script to recreate them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from modelswap.text import load_sentences, save_text_checkpoint

DEFAULTS = dict(
    train_data="data/sst2/train.jsonl",
    val_data="data/sst2/validation.jsonl",
    epochs=2,
    lr=2e-5,
    batch_size=16,
    max_len=64,
    seed=42,
)


def accuracy(model, tokenizer, sentences, labels, max_len, batch_size=64, device=None) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            enc = tokenizer(sentences[start:start + batch_size], padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            preds = model(**enc).logits.argmax(dim=-1).cpu().numpy()
            correct += int((preds == labels[start:start + batch_size]).sum())
    return correct / len(sentences)


def train(base: str, out: str | Path, log_out: str | Path | None = None, **overrides) -> dict:
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    model.to(device)

    train_sents, train_labels = load_sentences(cfg["train_data"])
    val_sents, val_labels = load_sentences(cfg["val_data"])

    steps_per_epoch = math.ceil(len(train_sents) / cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    loss_history = []
    model.train()
    for ep in range(cfg["epochs"]):
        order = np.random.permutation(len(train_sents))
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            enc = tokenizer([train_sents[i] for i in idx], padding=True, truncation=True,
                            max_length=cfg["max_len"], return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            labels_tensor = torch.from_numpy(train_labels[idx]).to(device)
            loss = model(**enc, labels=labels_tensor).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_history.append(round(float(loss), 6))

    val_acc = accuracy(model, tokenizer, val_sents, val_labels, cfg["max_len"], device=device)
    model.to("cpu")
    save_text_checkpoint(model, base, out)

    summary = {
        "base_model": base,
        "output": str(out),
        "train_examples": len(train_sents),
        "validation_examples": len(val_sents),
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "max_len": cfg["max_len"],
        "seed": cfg["seed"],
        "validation_accuracy": val_acc,
        "loss_history": loss_history,
    }
    if log_out is not None:
        Path(log_out).parent.mkdir(parents=True, exist_ok=True)
        Path(log_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"Fine-tuned {base} on SST-2 ({len(train_sents)} train, {cfg['epochs']} epochs)")
    print(f"Validation accuracy ({len(val_sents)} sentences): {val_acc * 100:.2f}%")
    print(f"Saved checkpoint: {out}")
    print("=" * 66)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a base encoder on SST-2.")
    parser.add_argument("--base", required=True, help="Hugging Face model id, e.g. bert-base-uncased")
    parser.add_argument("--out", required=True, help="Checkpoint path, e.g. models/text/bert_base_uncased.pt")
    parser.add_argument("--log-out", default=None, help="Optional JSON training log path")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = parser.parse_args()
    train(args.base, args.out, args.log_out, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, max_len=args.max_len, seed=args.seed)


if __name__ == "__main__":
    main()
