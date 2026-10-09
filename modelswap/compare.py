"""
Model Comparison and Architecture Verification Engine.

Evaluates Model A (Current) vs Model B (Candidate) on:
- Accuracy and cross-entropy loss
- Single-request latency (p50 and p95 in ms) and batch throughput
- Paired prediction agreement & disagreement analysis
- Architecture signature validation against registry
- Automated eligibility verdict generation (PASS / FAIL)
- Robust forward-pass smoke testing for apply/rollback safety
"""

import sys
from pathlib import Path
import json
import time
import argparse
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelswap.arch import MLP, load_checkpoint, architecture_signature

# Used when models.yaml does not set thresholds for a role. models.yaml is the source of truth.
DEFAULT_THRESHOLDS = {"min_accuracy": 0.85, "max_latency_ms": 5.0}

def set_deterministic_threads() -> None:
    """
    Set PyTorch intra-op parallelism to 1 thread for stable, jitter-free
    single-core latency benchmarking across runs.
    """
    # torch.set_num_threads(1) ensures single-threaded CPU execution
    # to avoid thread context switching artifacts in micro-benchmarks.
    torch.set_num_threads(1)

def evaluate(
    model: MLP,
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    n_latency_runs: int = 200
) -> dict[str, Any]:
    """
    Evaluates an MLP model on accuracy, cross-entropy loss, and latency.

    Timing methodology:
    - Sets torch.set_num_threads(1) for stable timings.
    - Runs 20 warm-up forward passes on a single sample (batch size = 1).
    - Measures n_latency_runs individual single-sample inferences using time.perf_counter().
    - Computes p50 and p95 latency in milliseconds.
    - Measures batch throughput (samples / second) over full dataset X.
    """
    set_deterministic_threads()
    model.eval()

    if isinstance(X, np.ndarray):
        X_t = torch.from_numpy(X.astype(np.float32))
    else:
        X_t = X.float()

    if isinstance(y, np.ndarray):
        y_t = torch.from_numpy(y.astype(np.int64))
    else:
        y_t = y.long()

    criterion = nn.CrossEntropyLoss()

    # 1. Full batch accuracy & loss
    with torch.no_grad():
        logits = model(X_t)
        loss = float(criterion(logits, y_t).item())
        preds = torch.argmax(logits, dim=-1)
        correct_count = int((preds == y_t).sum().item())
        total_count = int(y_t.shape[0])
        accuracy = float(correct_count / total_count)

    # 2. Single-request latency benchmark (batch size = 1)
    sample = X_t[0:1]
    with torch.no_grad():
        # 20 warm-up runs
        for _ in range(20):
            _ = model(sample)

        # n_latency_runs timed iterations
        times_ms = []
        for _ in range(n_latency_runs):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    latency_p50 = float(np.percentile(times_ms, 50))
    latency_p95 = float(np.percentile(times_ms, 95))
    latency_mean = float(np.mean(times_ms))

    # 3. Batch throughput
    with torch.no_grad():
        # Warmup batch
        for _ in range(3):
            _ = model(X_t)

        t_batch_start = time.perf_counter()
        n_batch_loops = 10
        for _ in range(n_batch_loops):
            _ = model(X_t)
        t_batch_end = time.perf_counter()

    batch_duration = t_batch_end - t_batch_start
    total_processed = total_count * n_batch_loops
    throughput = float(total_processed / batch_duration) if batch_duration > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": loss,
        "correct_count": correct_count,
        "total_count": total_count,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "latency_mean_ms": latency_mean,
        "throughput_samples_per_sec": throughput,
        "predictions": preds.cpu().numpy().tolist(),
    }

def verify_architecture(
    path_a: str | Path,
    path_b: str | Path,
    registry_entry: dict[str, Any] | None = None
) -> tuple[bool, list[str]]:
    """
    Compares input_dim, hidden_dim, output_dim, activation from architecture_signature,
    and against input_dim/output_dim in registry_entry (models.yaml).
    Returns (ok, list_of_mismatches).
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    mismatches: list[str] = []

    try:
        sig_a = architecture_signature(path_a)
    except Exception as e:
        return False, [f"Failed to read signature from Model A ({path_a}): {e}"]

    try:
        sig_b = architecture_signature(path_b)
    except Exception as e:
        return False, [f"Failed to read signature from Model B ({path_b}): {e}"]

    # Compare Model A vs Model B
    for attr in ["architecture", "input_dim", "hidden_dim", "output_dim", "activation"]:
        val_a = sig_a.get(attr)
        val_b = sig_b.get(attr)
        if val_a != val_b:
            mismatches.append(f"{attr}: {val_a} vs {val_b}")

    # Compare against registry expectations if provided
    if registry_entry:
        if "input_dim" in registry_entry and sig_a.get("input_dim") != registry_entry["input_dim"]:
            mismatches.append(
                f"registry input_dim: {registry_entry['input_dim']} vs model {sig_a.get('input_dim')}"
            )
        if "output_dim" in registry_entry and sig_a.get("output_dim") != registry_entry["output_dim"]:
            mismatches.append(
                f"registry output_dim: {registry_entry['output_dim']} vs model {sig_a.get('output_dim')}"
            )
        if "architecture" in registry_entry and sig_a.get("architecture") != registry_entry["architecture"]:
            mismatches.append(
                f"registry architecture: {registry_entry['architecture']} vs model {sig_a.get('architecture')}"
            )

    ok = len(mismatches) == 0
    return ok, mismatches

def smoke_test(model_path: str | Path) -> tuple[bool, str]:
    """
    Executes a runtime forward-pass smoke test on model_path.
    Stable import path for Arjun's swap/rollback engine:
        from modelswap.compare import smoke_test

    Validation criteria:
    - Checkpoint loads cleanly with load_checkpoint.
    - Forward pass runs without error on dummy zero batch (shape 1 x input_dim).
    - Output tensor has exact shape (1, output_dim).
    - Output contains only finite numerical values (no NaN or Inf).
    - Softmax probabilities sum to 1.0 (within 1e-5 tolerance).
    """
    path = Path(model_path)
    if not path.exists():
        return False, f"Smoke test failed: model file does not exist at {path}"

    try:
        model = load_checkpoint(path)
    except Exception as e:
        return False, f"Smoke test failed: could not load checkpoint ({e})"

    try:
        dummy = torch.zeros((1, model.input_dim), dtype=torch.float32)
        with torch.no_grad():
            logits = model(dummy)

        # Shape check
        expected_shape = (1, model.output_dim)
        if logits.shape != expected_shape:
            return False, f"Smoke test failed: output shape {logits.shape} != expected {expected_shape}"

        # Finite value check
        if not torch.isfinite(logits).all():
            return False, "Smoke test failed: output contains non-finite values (NaN/Inf)"

        # Softmax probability sum check
        probs = torch.softmax(logits, dim=-1)
        prob_sum = float(probs.sum().item())
        if not np.isclose(prob_sum, 1.0, atol=1e-5):
            return False, f"Smoke test failed: softmax probabilities sum to {prob_sum}, expected 1.0"

        return True, "Smoke test passed: valid output shape, finite values, and normalized probabilities."

    except Exception as e:
        return False, f"Smoke test failed during forward execution: {e}"

def verdict(result: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Evaluates swap eligibility against thresholds from models.yaml.
    Conditions for PASS:
    1. architecture_ok is True
    2. Candidate accuracy >= Current accuracy (no regression)
    3. Candidate accuracy >= min_accuracy threshold
    4. Candidate p95 latency <= max_latency_ms threshold
    """
    reasons: list[str] = []

    if not result.get("architecture_ok", False):
        mismatches = result.get("architecture_mismatches", ["Architecture mismatch"])
        return "FAIL", [f"Architecture check failed: {', '.join(mismatches)}"]

    min_accuracy = float(thresholds.get("min_accuracy", DEFAULT_THRESHOLDS["min_accuracy"]))
    max_latency_ms = float(thresholds.get("max_latency_ms", DEFAULT_THRESHOLDS["max_latency_ms"]))

    acc_a = result["current"]["accuracy"]
    acc_b = result["candidate"]["accuracy"]
    delta_acc = result["deltas"]["accuracy"]
    latency_b_p95 = result["candidate"]["latency_p95_ms"]

    passed = True

    # Check 1: Regression check
    if acc_b < acc_a:
        passed = False
        reasons.append(
            f"Candidate accuracy ({acc_b * 100:.2f}%) regressed below current accuracy ({acc_a * 100:.2f}%)."
        )
    else:
        reasons.append(
            f"Candidate accuracy ({acc_b * 100:.2f}%) improved on current accuracy ({acc_a * 100:.2f}%) by {delta_acc * 100:+.2f}%."
        )

    # Check 2: Minimum accuracy threshold
    if acc_b < min_accuracy:
        passed = False
        reasons.append(
            f"Candidate accuracy ({acc_b * 100:.2f}%) is below minimum threshold ({min_accuracy * 100:.2f}%)."
        )
    else:
        reasons.append(
            f"Candidate accuracy ({acc_b * 100:.2f}%) meets minimum threshold ({min_accuracy * 100:.2f}%)."
        )

    # Check 3: Maximum latency threshold
    if latency_b_p95 > max_latency_ms:
        passed = False
        reasons.append(
            f"Candidate p95 latency ({latency_b_p95:.4f}ms) exceeds maximum threshold ({max_latency_ms:.4f}ms)."
        )
    else:
        reasons.append(
            f"Candidate p95 latency ({latency_b_p95:.4f}ms) is within threshold ({max_latency_ms:.4f}ms)."
        )

    status = "PASS" if passed else "FAIL"
    return status, reasons

def compare(
    role_or_paths: str | tuple[str | Path, str | Path] | list[str | Path],
    data_path: str | Path = "data/test.npz",
    models_yaml: str | Path = "models.yaml",
    n_latency_runs: int = 200,
    registry_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main comparison entrypoint. Resolves model paths from role or arguments,
    verifies architecture, evaluates both models, and computes paired agreement.

    Pass registry_entry (the role's dict from models.yaml) to skip the path-based
    lookup in tuple mode.
    """
    thresholds = dict(DEFAULT_THRESHOLDS)
    if registry_entry is not None:
        thresholds.update(registry_entry.get("thresholds") or {})

    yaml_file = Path(models_yaml)
    if isinstance(role_or_paths, str) and not Path(role_or_paths).exists():
        # Treat as role name
        role = role_or_paths
        if not yaml_file.exists():
            raise FileNotFoundError(f"models.yaml not found at {yaml_file}")
        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        roles = cfg.get("roles", {})
        if role not in roles:
            raise KeyError(f"Role '{role}' not found in {yaml_file}. Available: {list(roles.keys())}")
        registry_entry = roles[role]
        path_a = registry_entry.get("current", "models/model_a.pt")
        path_b = registry_entry.get("candidate", "models/model_b.pt")
        if "thresholds" in registry_entry:
            thresholds.update(registry_entry["thresholds"])
    elif isinstance(role_or_paths, (tuple, list)) and len(role_or_paths) == 2:
        path_a, path_b = role_or_paths
        if registry_entry is None and yaml_file.exists():
            with open(yaml_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                roles = cfg.get("roles", {})
                for _, rcfg in roles.items():
                    if rcfg.get("current") == str(path_a) or rcfg.get("candidate") == str(path_b):
                        registry_entry = rcfg
                        if "thresholds" in rcfg:
                            thresholds.update(rcfg["thresholds"])
                        break
    else:
        path_a = "models/model_a.pt"
        path_b = "models/model_b.pt"

    path_a = str(path_a)
    path_b = str(path_b)

    # 1. Architecture Verification Gate
    arch_ok, mismatches = verify_architecture(path_a, path_b, registry_entry)
    if not arch_ok:
        res = {
            "current_path": path_a,
            "candidate_path": path_b,
            "architecture_ok": False,
            "architecture_mismatches": mismatches,
            "verdict": "FAIL",
            "verdict_reasons": [f"Architecture check failed: {', '.join(mismatches)}"],
            "current": None,
            "candidate": None,
            "deltas": None,
            "paired_agreement": None,
        }
        return res

    # 2. Load evaluation dataset
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {data_path}")

    if data_path.suffix == ".npz":
        npz_data = np.load(data_path)
        X = npz_data["X"]
        y = npz_data["y"]
    elif data_path.suffix == ".csv":
        raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
        X = raw[:, :2]
        y = raw[:, 2].astype(np.int64)
    else:
        raise ValueError(f"Unsupported data format {data_path.suffix}, expected .npz or .csv")

    # 3. Load models & evaluate
    model_a = load_checkpoint(path_a)
    model_b = load_checkpoint(path_b)

    eval_a = evaluate(model_a, X, y, n_latency_runs=n_latency_runs)
    eval_b = evaluate(model_b, X, y, n_latency_runs=n_latency_runs)

    # 4. Compute Deltas
    deltas = {
        "accuracy": float(eval_b["accuracy"] - eval_a["accuracy"]),
        "loss": float(eval_b["loss"] - eval_a["loss"]),
        "latency_p50_ms": float(eval_b["latency_p50_ms"] - eval_a["latency_p50_ms"]),
        "latency_p95_ms": float(eval_b["latency_p95_ms"] - eval_a["latency_p95_ms"]),
        "latency_mean_ms": float(eval_b["latency_mean_ms"] - eval_a["latency_mean_ms"]),
        "throughput_samples_per_sec": float(eval_b["throughput_samples_per_sec"] - eval_a["throughput_samples_per_sec"]),
    }

    # 5. Paired Prediction Agreement
    preds_a = np.array(eval_a["predictions"])
    preds_b = np.array(eval_b["predictions"])
    y_true = np.array(y)

    diff_mask = (preds_a != preds_b)
    diff_count = int(diff_mask.sum())
    b_better = int(((preds_b == y_true) & (preds_a != y_true)).sum())
    a_better = int(((preds_a == y_true) & (preds_b != y_true)).sum())
    both_correct = int(((preds_a == y_true) & (preds_b == y_true)).sum())
    both_wrong = int(((preds_a != y_true) & (preds_b != y_true)).sum())

    paired_agreement = {
        "differ_count": diff_count,
        "differ_ratio": float(diff_count / len(y_true)) if len(y_true) > 0 else 0.0,
        "candidate_correct_current_wrong": b_better,
        "current_correct_candidate_wrong": a_better,
        "net_gain_samples": b_better - a_better,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }

    result = {
        "current_path": path_a,
        "candidate_path": path_b,
        "architecture_ok": True,
        "architecture_mismatches": [],
        "current": {
            "accuracy": eval_a["accuracy"],
            "loss": eval_a["loss"],
            "latency_p50_ms": eval_a["latency_p50_ms"],
            "latency_p95_ms": eval_a["latency_p95_ms"],
            "latency_mean_ms": eval_a["latency_mean_ms"],
            "throughput_samples_per_sec": eval_a["throughput_samples_per_sec"],
        },
        "candidate": {
            "accuracy": eval_b["accuracy"],
            "loss": eval_b["loss"],
            "latency_p50_ms": eval_b["latency_p50_ms"],
            "latency_p95_ms": eval_b["latency_p95_ms"],
            "latency_mean_ms": eval_b["latency_mean_ms"],
            "throughput_samples_per_sec": eval_b["throughput_samples_per_sec"],
        },
        "deltas": deltas,
        "paired_agreement": paired_agreement,
        "thresholds": thresholds,
    }

    # 6. Verdict
    status, reasons = verdict(result, thresholds)
    result["verdict"] = status
    result["verdict_reasons"] = reasons

    return result

def print_comparison(result: dict[str, Any]) -> None:
    """
    Renders an ASCII table of comparison metrics, paired analysis, and verdict for CLI.
    """
    print("==================================================================")
    print("                 ML MODEL SWAP: COMPARISON REPORT                 ")
    print("==================================================================")
    print(f"Current Model   : {result['current_path']}")
    print(f"Candidate Model : {result['candidate_path']}")
    print("------------------------------------------------------------------")

    if not result.get("architecture_ok", False):
        print("ARCHITECTURE CHECK : [FAIL]")
        for m in result.get("architecture_mismatches", []):
            print(f"  - {m}")
        print("------------------------------------------------------------------")
        print("VERDICT            : [FAIL] (Architecture mismatch prevents evaluation)")
        print("==================================================================")
        return

    curr = result["current"]
    cand = result["candidate"]
    delts = result["deltas"]
    pair = result["paired_agreement"]

    print(f"{'Metric':<22} | {'Current (A)':<12} | {'Candidate (B)':<13} | {'Delta (B - A)':<12}")
    print("------------------------------------------------------------------")
    print(f"{'Accuracy':<22} | {curr['accuracy']*100:>10.2f}% | {cand['accuracy']*100:>11.2f}% | {delts['accuracy']*100:>+10.2f}%")
    print(f"{'Cross-Entropy Loss':<22} | {curr['loss']:>11.4f} | {cand['loss']:>12.4f} | {delts['loss']:>+11.4f}")
    print(f"{'Latency p50 (ms)':<22} | {curr['latency_p50_ms']:>9.4f}ms | {cand['latency_p50_ms']:>10.4f}ms | {delts['latency_p50_ms']:>+9.4f}ms")
    print(f"{'Latency p95 (ms)':<22} | {curr['latency_p95_ms']:>9.4f}ms | {cand['latency_p95_ms']:>10.4f}ms | {delts['latency_p95_ms']:>+9.4f}ms")
    print(f"{'Throughput (samp/s)':<22} | {curr['throughput_samples_per_sec']:>11.0f} | {cand['throughput_samples_per_sec']:>12.0f} | {delts['throughput_samples_per_sec']:>+11.0f}")
    print("------------------------------------------------------------------")
    print("PAIRED AGREEMENT ANALYSIS (Test Set):")
    print(f"  Total test points where A and B disagree : {pair['differ_count']} ({pair['differ_ratio']*100:.1f}%)")
    print(f"  Cases where Candidate (B) is right & A is wrong : {pair['candidate_correct_current_wrong']}")
    print(f"  Cases where Current (A) is right & B is wrong   : {pair['current_correct_candidate_wrong']}")
    print(f"  Net correct classification gain by B            : {pair['net_gain_samples']:+d} samples")
    print("------------------------------------------------------------------")
    status_icon = "[PASS]" if result["verdict"] == "PASS" else "[FAIL]"
    print(f"SWAP ELIGIBILITY VERDICT : {status_icon}")
    for reason in result.get("verdict_reasons", []):
        print(f"  * {reason}")
    print("==================================================================")

def write_json(result: dict[str, Any], path: str | Path) -> None:
    """
    Saves serializable comparison result to JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and compare Model A vs Model B.")
    parser.add_argument("--role", default="classifier", help="Role name in models.yaml (default 'classifier')")
    parser.add_argument("--model-a", default=None, help="Explicit path to Model A (overrides role)")
    parser.add_argument("--model-b", default=None, help="Explicit path to Model B (overrides role)")
    parser.add_argument("--data", default="data/test.npz", help="Path to test data (.npz or .csv)")
    parser.add_argument("--models-yaml", default="models.yaml", help="Path to models.yaml")
    parser.add_argument("--json-out", default=None, help="Optional path to output JSON results")
    args = parser.parse_args()

    if args.model_a and args.model_b:
        role_or_paths = (args.model_a, args.model_b)
    else:
        role_or_paths = args.role

    try:
        res = compare(role_or_paths, data_path=args.data, models_yaml=args.models_yaml)
        print_comparison(res)
        if args.json_out:
            write_json(res, args.json_out)
        return 0 if res["verdict"] == "PASS" else 1
    except Exception as e:
        print(f"Comparison error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
