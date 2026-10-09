"""Text classification role: fine-tuned BERT / RoBERTa checkpoints, checks and evaluation.

A text checkpoint is one .pt file holding the model's config and state_dict,
plus the base model name (used to load its tokenizer). Each model is built in
its own class (BertForSequenceClassification, RobertaForSequenceClassification),
so the registry entry for this role sets cross_architecture: true.

compare_text() returns a report with the same shape as modelswap.compare.compare(),
so the same verdict() rule applies to both roles.
"""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

FORMAT = "hf-seqcls-v1"
MAX_LEN = 64
SMOKE_SENTENCE = "a short smoke test sentence"


def save_text_checkpoint(model, base_model: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": FORMAT,
        "base_model": base_model,
        "model_type": model.config.model_type,
        "architecture": f"{model.config.model_type}_seqcls",
        "num_labels": int(model.config.num_labels),
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def _read_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} text checkpoint")
    return checkpoint


def read_text_meta(path: str | Path) -> dict:
    """Checkpoint metadata without the weights."""
    checkpoint = _read_checkpoint(path)
    return {key: value for key, value in checkpoint.items() if key != "state_dict"}


def load_text_checkpoint(path: str | Path):
    """Return (model, tokenizer, meta) in eval mode."""
    checkpoint = _read_checkpoint(path)
    config = AutoConfig.from_dict(checkpoint["config"])
    model = AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["base_model"])
    meta = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, tokenizer, meta


def smoke_test_text(model_path: str | Path) -> tuple[bool, str]:
    """Forward pass on one sentence: logits shape (1, num_labels), finite, softmax sums to 1."""
    try:
        model, tokenizer, meta = load_text_checkpoint(model_path)
    except Exception as exc:
        return False, f"Smoke test failed: could not load checkpoint ({exc})"
    try:
        enc = tokenizer(SMOKE_SENTENCE, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        expected = (1, int(meta["num_labels"]))
        if tuple(logits.shape) != expected:
            return False, f"Smoke test failed: output shape {tuple(logits.shape)} != expected {expected}"
        if not torch.isfinite(logits).all():
            return False, "Smoke test failed: output contains non-finite values"
        prob_sum = float(torch.softmax(logits, dim=-1).sum())
        if not math.isclose(prob_sum, 1.0, abs_tol=1e-5):
            return False, f"Smoke test failed: softmax probabilities sum to {prob_sum}"
        return True, "Smoke test passed: valid output shape, finite values, and normalized probabilities."
    except Exception as exc:
        return False, f"Smoke test failed during forward execution: {exc}"


def verify_text_pair(candidate: str, current: str, entry: dict) -> tuple[bool, str]:
    """Pre-check for apply: the candidate must load and match the head size of the current model.

    Different architectures are allowed only when the registry entry sets cross_architecture: true.
    """
    try:
        cand = read_text_meta(candidate)
        cur = read_text_meta(current)
    except Exception as exc:
        return False, f"could not read checkpoint metadata: {exc}"
    if cand["num_labels"] != cur["num_labels"]:
        return False, f"num_labels differs: {cur['num_labels']} vs {cand['num_labels']}"
    if cand["architecture"] != cur["architecture"] and not entry.get("cross_architecture", False):
        return False, f"architecture differs: {cur['architecture']} vs {cand['architecture']}"
    return True, f"{cur['architecture']} -> {cand['architecture']}, num_labels={cand['num_labels']}"


def load_sentences(data_path: str | Path) -> tuple[list[str], np.ndarray]:
    sentences, labels = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sentences.append(row["sentence"])
            labels.append(int(row["label"]))
    return sentences, np.array(labels, dtype=np.int64)


def evaluate_text(model, tokenizer, sentences: list[str], labels: np.ndarray,
                  n_latency_runs: int = 50, batch_size: int = 32) -> dict:
    """Accuracy, loss and single-sentence latency, measured the same way for every model."""
    logits_parts = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            enc = tokenizer(sentences[start:start + batch_size], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            logits_parts.append(model(**enc).logits)
    logits = torch.cat(logits_parts)
    y = torch.from_numpy(labels)
    preds = logits.argmax(dim=-1)
    accuracy = float((preds == y).float().mean())
    loss = float(F.cross_entropy(logits, y))

    times_ms = []
    with torch.no_grad():
        for i in range(5):  # warm-up
            model(**tokenizer(sentences[i], truncation=True, max_length=MAX_LEN, return_tensors="pt"))
        for i in range(n_latency_runs):
            enc = tokenizer(sentences[i % len(sentences)], truncation=True, max_length=MAX_LEN, return_tensors="pt")
            t0 = time.perf_counter()
            model(**enc)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "accuracy": accuracy,
        "loss": loss,
        "latency_p50_ms": float(np.percentile(times_ms, 50)),
        "latency_p95_ms": float(np.percentile(times_ms, 95)),
        "latency_mean_ms": float(np.mean(times_ms)),
        "predictions": preds.numpy().tolist(),
    }


def compare_text(current_path: str, candidate_path: str, data_path: str | Path,
                 thresholds: dict, n_latency_runs: int = 50) -> dict:
    """Compare current (A) and candidate (B) on the validation sentences. Returns the compare report shape."""
    from modelswap.compare import verdict

    cur_meta = read_text_meta(current_path)
    cand_meta = read_text_meta(candidate_path)
    mismatches = []
    if cur_meta["num_labels"] != cand_meta["num_labels"]:
        mismatches.append(f"num_labels: {cur_meta['num_labels']} vs {cand_meta['num_labels']}")
    if mismatches:
        return {
            "current_path": str(current_path),
            "candidate_path": str(candidate_path),
            "architecture_ok": False,
            "architecture_mismatches": mismatches,
            "verdict": "FAIL",
            "verdict_reasons": [f"Architecture check failed: {', '.join(mismatches)}"],
            "current": None,
            "candidate": None,
            "deltas": None,
            "paired_agreement": None,
        }

    sentences, labels = load_sentences(data_path)

    model_a, tok_a, _ = load_text_checkpoint(current_path)
    eval_a = evaluate_text(model_a, tok_a, sentences, labels, n_latency_runs)
    del model_a, tok_a
    gc.collect()

    model_b, tok_b, _ = load_text_checkpoint(candidate_path)
    eval_b = evaluate_text(model_b, tok_b, sentences, labels, n_latency_runs)
    del model_b, tok_b
    gc.collect()

    preds_a = np.array(eval_a.pop("predictions"))
    preds_b = np.array(eval_b.pop("predictions"))
    b_better = int(((preds_b == labels) & (preds_a != labels)).sum())
    a_better = int(((preds_a == labels) & (preds_b != labels)).sum())
    diff_count = int((preds_a != preds_b).sum())

    keys = ("accuracy", "loss", "latency_p50_ms", "latency_p95_ms", "latency_mean_ms")
    result = {
        "current_path": str(current_path),
        "candidate_path": str(candidate_path),
        "architecture_ok": True,
        "architecture_mismatches": [],
        "current": {k: eval_a[k] for k in keys},
        "candidate": {k: eval_b[k] for k in keys},
        "deltas": {k: float(eval_b[k] - eval_a[k]) for k in keys},
        "paired_agreement": {
            "differ_count": diff_count,
            "differ_ratio": float(diff_count / len(labels)),
            "candidate_correct_current_wrong": b_better,
            "current_correct_candidate_wrong": a_better,
            "net_gain_samples": b_better - a_better,
        },
        "thresholds": dict(thresholds),
    }
    status, reasons = verdict(result, thresholds)
    result["verdict"] = status
    result["verdict_reasons"] = reasons
    return result
