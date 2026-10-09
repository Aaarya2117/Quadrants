"""Prepare the SST-2 sentiment data used by the text role (bert-base-uncased vs roberta-base).

Writes JSON lines with {"sentence": ..., "label": 0|1}:
- data/sst2/train.jsonl       fixed random subset of the GLUE SST-2 train split (seed 42)
- data/sst2/validation.jsonl  the full GLUE SST-2 validation split (872 sentences)

The GLUE test split has no public labels, so validation is the evaluation set.
It is never used for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "stanfordnlp/sst2"


def write_jsonl(rows, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"sentence": row["sentence"], "label": int(row["label"])}) + "\n")
    return len(rows)


def prepare(train_size: int = 4000, seed: int = 42, output_dir: str | Path = "data/sst2") -> dict:
    output_dir = Path(output_dir)
    ds = load_dataset(DATASET)
    train = ds["train"].shuffle(seed=seed).select(range(train_size))
    validation = ds["validation"]

    n_train = write_jsonl(train, output_dir / "train.jsonl")
    n_val = write_jsonl(validation, output_dir / "validation.jsonl")
    summary = {"train": n_train, "validation": n_val, "seed": seed, "source": DATASET}
    print(f"SST-2 prepared: train={n_train} (subset, seed={seed}), validation={n_val} -> {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SST-2 train subset and validation split.")
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/sst2")
    args = parser.parse_args()
    prepare(args.train_size, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
