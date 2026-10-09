from pathlib import Path
import pytest
import torch
import numpy as np

from modelswap.arch import MLP, save_checkpoint
from modelswap.compare import (
    evaluate,
    verify_architecture,
    smoke_test,
    verdict,
    compare,
)

def test_matching_architectures_pass(tmp_path: Path):
    m1 = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    m2 = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    p1 = tmp_path / "m1.pt"
    p2 = tmp_path / "m2.pt"
    save_checkpoint(m1, p1)
    save_checkpoint(m2, p2)

    ok, mismatches = verify_architecture(p1, p2, registry_entry={"input_dim": 2, "output_dim": 2})
    assert ok is True
    assert len(mismatches) == 0

def test_mismatched_hidden_dim_fails_with_reason(tmp_path: Path):
    m1 = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    m2 = MLP(input_dim=2, hidden_dim=32, output_dim=2)
    p1 = tmp_path / "m1.pt"
    p2 = tmp_path / "m2.pt"
    save_checkpoint(m1, p1)
    save_checkpoint(m2, p2)

    ok, mismatches = verify_architecture(p1, p2)
    assert ok is False
    assert any("hidden_dim: 16 vs 32" in m for m in mismatches)

    # Compare must return FAIL immediately without computing metrics
    res = compare((p1, p2), data_path="data/test.npz")
    assert res["architecture_ok"] is False
    assert res["verdict"] == "FAIL"
    assert res["current"] is None

def test_corrupted_checkpoint_fails_smoke_test(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pt"
    # Write garbage data
    with open(corrupt_path, "wb") as f:
        f.write(b"not a valid pytorch checkpoint")

    ok, msg = smoke_test(corrupt_path)
    assert ok is False
    assert "Smoke test failed" in msg

    # Non-existent path
    ok_missing, msg_missing = smoke_test(tmp_path / "missing.pt")
    assert ok_missing is False
    assert "does not exist" in msg_missing

def test_smoke_test_passes_on_valid_checkpoints():
    ok_a, msg_a = smoke_test("models/model_a.pt")
    ok_b, msg_b = smoke_test("models/model_b.pt")
    assert ok_a is True
    assert "passed" in msg_a
    assert ok_b is True
    assert "passed" in msg_b

def test_worse_candidate_gets_fail():
    synthetic_result = {
        "architecture_ok": True,
        "current": {"accuracy": 0.90, "latency_p95_ms": 0.02},
        "candidate": {"accuracy": 0.82, "latency_p95_ms": 0.02},
        "deltas": {"accuracy": -0.08},
    }
    thresholds = {"min_accuracy": 0.80, "max_latency_ms": 5.0}
    status, reasons = verdict(synthetic_result, thresholds)
    assert status == "FAIL"
    assert any("regressed below current accuracy" in r for r in reasons)

def test_latency_exceeded_gets_fail():
    synthetic_result = {
        "architecture_ok": True,
        "current": {"accuracy": 0.85, "latency_p95_ms": 0.02},
        "candidate": {"accuracy": 0.92, "latency_p95_ms": 6.50},
        "deltas": {"accuracy": +0.07},
    }
    thresholds = {"min_accuracy": 0.80, "max_latency_ms": 5.0}
    status, reasons = verdict(synthetic_result, thresholds)
    assert status == "FAIL"
    assert any("exceeds maximum threshold" in r for r in reasons)
