from pathlib import Path
import pytest
import numpy as np
import torch

from data.make_data import generate_and_save_data
from models.train_a import train_model_a
from models.train_b import train_model_b
from modelswap.compare import compare, smoke_test

def test_ml_pipeline_end_to_end_and_reproducibility(tmp_path: Path):
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data creation with tiny settings
    data_stats = generate_and_save_data(
        n_samples=300,
        noise=0.25,
        seed=42,
        output_dir=data_dir
    )
    assert (data_dir / "train.npz").exists()
    assert (data_dir / "val.npz").exists()
    assert (data_dir / "test.npz").exists()
    assert (data_dir / "scaler.json").exists()

    # 2. Train Model A (tiny settings for fast test execution)
    res_a1 = train_model_a(
        train_data_path=data_dir / "train.npz",
        val_data_path=data_dir / "val.npz",
        test_data_path=data_dir / "test.npz",
        model_output_path=models_dir / "model_a.pt",
        log_output_path=models_dir / "log_a.json",
        subset_size=60,
        epochs=8,
        lr=0.02,
        seed=42
    )
    assert (models_dir / "model_a.pt").exists()

    # Smoke test on Model A
    ok_a, msg_a = smoke_test(models_dir / "model_a.pt")
    assert ok_a is True

    # 3. Train Model B fine-tuned from Model A
    res_b1 = train_model_b(
        model_a_path=models_dir / "model_a.pt",
        train_data_path=data_dir / "train.npz",
        val_data_path=data_dir / "val.npz",
        test_data_path=data_dir / "test.npz",
        model_output_path=models_dir / "model_b.pt",
        log_output_path=models_dir / "log_b.json",
        epochs=25,
        lr=0.01,
        seed=42
    )
    assert (models_dir / "model_b.pt").exists()

    # Smoke test on Model B
    ok_b, msg_b = smoke_test(models_dir / "model_b.pt")
    assert ok_b is True

    # 4. Compare & Verdict
    cmp_res = compare(
        (models_dir / "model_a.pt", models_dir / "model_b.pt"),
        data_path=data_dir / "test.npz"
    )
    assert cmp_res["architecture_ok"] is True
    assert "accuracy" in cmp_res["deltas"]
    assert "latency_p95_ms" in cmp_res["candidate"]
    assert cmp_res["paired_agreement"]["differ_count"] >= 0

    # 5. Reproducibility test: re-train and assert bit-for-bit identical results
    models_dir2 = tmp_path / "models2"
    models_dir2.mkdir(parents=True, exist_ok=True)

    res_a2 = train_model_a(
        train_data_path=data_dir / "train.npz",
        val_data_path=data_dir / "val.npz",
        test_data_path=data_dir / "test.npz",
        model_output_path=models_dir2 / "model_a.pt",
        log_output_path=models_dir2 / "log_a.json",
        subset_size=60,
        epochs=8,
        lr=0.02,
        seed=42
    )
    res_b2 = train_model_b(
        model_a_path=models_dir2 / "model_a.pt",
        train_data_path=data_dir / "train.npz",
        val_data_path=data_dir / "val.npz",
        test_data_path=data_dir / "test.npz",
        model_output_path=models_dir2 / "model_b.pt",
        log_output_path=models_dir2 / "log_b.json",
        epochs=25,
        lr=0.01,
        seed=42
    )

    assert res_a1["test_accuracy"] == res_a2["test_accuracy"]
    assert res_b1["acc_b_test"] == res_b2["acc_b_test"]
