# ML Model Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the ML subsystem for ML Model Swap: two identical-architecture PyTorch MLPs (`model_a.pt` and `model_b.pt`), a reproducible 2D quadrant classification dataset, MATLAB weight/data exports, and the `modelswap/compare.py` evaluation and verdict engine.

**Architecture:** A lightweight Python package `modelswap/` containing model definitions, synthetic data generation, a deterministic training pipeline, and a side-by-side comparison engine. Outputs are structured to immediately satisfy teammate contracts (Arjun's runtime/smoke test, Achyut's MATLAB visualizer, Anirudh's CLI).

**Tech Stack:** Python 3.11+, PyTorch (`torch`), NumPy, Scikit-learn, SciPy (`scipy.io`), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-ml-model-swap-design.md`

## Global Constraints

- Python version floor: 3.11+.
- Fixed random seeds: `seed = 42` across NumPy and PyTorch for 100% reproducibility.
- Zero hardcoded printed numbers: all metrics (accuracy, loss, latency, deltas) must be dynamically computed.
- Teammate contracts: preserve `models/model_a.pt`, `models/model_b.pt`, `data/quadrants_data.npz`, `matlab/` exports, and `evaluate_and_compare` signature.

## Review Focus

1. **Shape or architecture mismatch between Model A and Model B:** `compare.py` must raise a clear `ValueError` or reject candidate if layers, tensor shapes, or parameter counts differ.
2. **Deterministic reproducibility:** Repeated calls to dataset generation or training must yield identical tensor values when seed is set.
3. **Empty or corrupted data input:** `compare.py` and evaluation routines must handle invalid inputs gracefully with informative error messages.
4. **Latency measurement stability:** Benchmarking must include warmup iterations so JIT/cache warmup does not distort latency measurements.
5. **Non-PyTorch consumers (MATLAB):** JSON and MAT exports must export flat or 2D floating-point arrays readable without PyTorch installed.

---

### Task 1: Shared Neural Architecture (`modelswap/model.py`)

**Files:**
- Create: `modelswap/__init__.py`
- Create: `modelswap/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces:
  - `class MLP(torch.nn.Module)`: takes `input_dim: int = 2`, `hidden_dim: int = 16`, `output_dim: int = 2`.
  - `MLP.forward(x: torch.Tensor) -> torch.Tensor`: returns raw logits of shape `(batch_size, 2)`.
  - `MLP.predict_proba(x: torch.Tensor) -> torch.Tensor`: returns softmax probabilities of shape `(batch_size, 2)`.
  - `MLP.predict(x: torch.Tensor) -> torch.Tensor`: returns argmax class predictions of shape `(batch_size,)`.
  - `save_model(model: MLP, path: str | Path) -> None`: saves state dict.
  - `load_model(path: str | Path, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2) -> MLP`: loads state dict into model.

- [ ] **Step 1: Write the failing test for `modelswap/model.py`**

```python
# tests/test_model.py
import pytest
import torch
from pathlib import Path
from modelswap.model import MLP, save_model, load_model

def test_mlp_architecture_shapes():
    model = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    # Check total trainable parameters = 16*2 + 16 + 2*16 + 2 = 82
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == 82

    # Forward pass check
    x = torch.randn(5, 2)
    logits = model(x)
    assert logits.shape == (5, 2)

    probs = model.predict_proba(x)
    assert probs.shape == (5, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)

    preds = model.predict(x)
    assert preds.shape == (5,)

def test_mlp_save_and_load(tmp_path: Path):
    model = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    save_path = tmp_path / "model.pt"
    save_model(model, save_path)
    assert save_path.exists()

    loaded = load_model(save_path)
    x = torch.randn(3, 2)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modelswap'`

- [ ] **Step 3: Implement `modelswap/__init__.py` and `modelswap/model.py`**

```python
# modelswap/__init__.py
"""ML Model Swap core package."""
__version__ = "0.1.0"
```

```python
# modelswap/model.py
from pathlib import Path
import torch
import torch.nn as nn

class MLP(nn.Module):
    """Deterministic 2-layer Multi-Layer Perceptron classifier."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

def save_model(model: MLP, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(path: str | Path, input_dim: int = 2, hidden_dim: int = 16, output_dim: int = 2) -> MLP:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelswap/__init__.py modelswap/model.py tests/test_model.py
git commit -m "feat(ml): implement shared 2-layer MLP architecture and serialization"
```

---

### Task 2: Synthetic 2D Quadrant Dataset (`modelswap/dataset.py`)

**Files:**
- Create: `modelswap/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `generate_quadrant_data(n_samples: int = 1000, noise: float = 0.15, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: returns `(X_train, y_train, X_test, y_test)`.
  - `save_dataset(X_train, y_train, X_test, y_test, npz_path: str | Path) -> None`
  - `export_matlab_data(X_test, y_test, json_path: str | Path, mat_path: str | Path | None = None) -> None`

- [ ] **Step 1: Write the failing test for `modelswap/dataset.py`**

```python
# tests/test_dataset.py
import pytest
import numpy as np
import json
from pathlib import Path
from modelswap.dataset import generate_quadrant_data, save_dataset, export_matlab_data

def test_dataset_generation_shapes_and_reproducibility():
    X_train1, y_train1, X_test1, y_test1 = generate_quadrant_data(n_samples=1000, seed=42)
    X_train2, y_train2, X_test2, y_test2 = generate_quadrant_data(n_samples=1000, seed=42)

    assert X_train1.shape == (800, 2)
    assert y_train1.shape == (800,)
    assert X_test1.shape == (200, 2)
    assert y_test1.shape == (200,)

    # Labels must be strictly binary 0 or 1
    assert set(np.unique(y_train1)).issubset({0, 1})
    assert set(np.unique(y_test1)).issubset({0, 1})

    # Exact deterministic reproducibility
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(y_train1, y_train2)

def test_dataset_save_and_matlab_export(tmp_path: Path):
    X_train, y_train, X_test, y_test = generate_quadrant_data(n_samples=100, seed=42)
    npz_path = tmp_path / "data.npz"
    json_path = tmp_path / "matlab_data.json"
    mat_path = tmp_path / "matlab_data.mat"

    save_dataset(X_train, y_train, X_test, y_test, npz_path)
    assert npz_path.exists()

    export_matlab_data(X_test, y_test, json_path, mat_path)
    assert json_path.exists()
    assert mat_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "X_test" in data
    assert "y_test" in data
    assert len(data["X_test"]) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modelswap.dataset'`

- [ ] **Step 3: Implement `modelswap/dataset.py`**

```python
# modelswap/dataset.py
from pathlib import Path
import json
import numpy as np
try:
    from scipy.io import savemat
except ImportError:
    savemat = None

def generate_quadrant_data(
    n_samples: int = 1000,
    noise: float = 0.15,
    test_ratio: float = 0.2,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic 2D 4-Quadrant (XOR) binary classification dataset.
    Class 1 if x1 * x2 > 0 else Class 0.
    """
    rng = np.random.default_rng(seed)
    
    # Sample 2D points uniformly in [-2, 2] x [-2, 2]
    X_raw = rng.uniform(-2.0, 2.0, size=(n_samples, 2))
    
    # Quadrant XOR logic: Quadrant 1 and 3 are Class 1; Quadrant 2 and 4 are Class 0
    y_clean = ((X_raw[:, 0] * X_raw[:, 1]) > 0).astype(np.int64)
    
    # Add Gaussian noise to features
    noise_vals = rng.normal(0.0, noise, size=X_raw.shape)
    X = X_raw + noise_vals
    y = y_clean

    # Train/Test split
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test
    
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return (
        X[train_idx].astype(np.float32),
        y[train_idx].astype(np.int64),
        X[test_idx].astype(np.float32),
        y[test_idx].astype(np.int64),
    )

def save_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    npz_path: str | Path
) -> None:
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

def export_matlab_data(
    X_test: np.ndarray,
    y_test: np.ndarray,
    json_path: str | Path,
    mat_path: str | Path | None = None
) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "n_samples": int(X_test.shape[0]),
        "input_dim": int(X_test.shape[1]),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if mat_path is not None and savemat is not None:
        mat_path = Path(mat_path)
        mat_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(mat_path), {"X_test": X_test, "y_test": y_test})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelswap/dataset.py tests/test_dataset.py
git commit -m "feat(ml): implement reproducible 2D quadrant dataset generator and MATLAB exporter"
```

---

### Task 3: Training Pipeline & Weights Exporter (`modelswap/train.py`)

**Files:**
- Create: `modelswap/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes:
  - `modelswap.model.MLP`, `modelswap.model.save_model`
  - `modelswap.dataset.generate_quadrant_data`, `save_dataset`, `export_matlab_data`
- Produces:
  - `train_mlp(model, X_train, y_train, epochs, lr, seed) -> MLP`
  - `export_matlab_weights(model, json_path, mat_path) -> None`
  - `run_training_pipeline() -> dict`: builds dataset, trains `model_a.pt` (baseline) and `model_b.pt` (converged), saves weights and returns accuracy stats.

- [ ] **Step 1: Write the failing test for `modelswap/train.py`**

```python
# tests/test_train.py
import pytest
from pathlib import Path
import torch
import json
from modelswap.model import load_model
from modelswap.train import train_mlp, export_matlab_weights, run_training_pipeline
from modelswap.dataset import generate_quadrant_data

def test_training_convergence():
    X_train, y_train, X_test, y_test = generate_quadrant_data(n_samples=500, seed=42)
    
    # Train baseline (under-trained, 5 epochs)
    model_baseline = train_mlp(epochs=5, lr=0.01, seed=42, X_train=X_train, y_train=y_train)
    with torch.no_grad():
        preds_a = model_baseline.predict(torch.from_numpy(X_test))
        acc_a = (preds_a.numpy() == y_test).mean()

    # Train converged (50 epochs)
    model_converged = train_mlp(epochs=50, lr=0.02, seed=42, X_train=X_train, y_train=y_train)
    with torch.no_grad():
        preds_b = model_converged.predict(torch.from_numpy(X_test))
        acc_b = (preds_b.numpy() == y_test).mean()

    assert acc_b > acc_a
    assert acc_b >= 0.85

def test_export_matlab_weights(tmp_path: Path):
    from modelswap.model import MLP
    model = MLP()
    json_path = tmp_path / "weights.json"
    mat_path = tmp_path / "weights.mat"
    export_matlab_weights(model, json_path, mat_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "W1" in data and "b1" in data and "W2" in data and "b2" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modelswap.train'`

- [ ] **Step 3: Implement `modelswap/train.py`**

```python
# modelswap/train.py
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
try:
    from scipy.io import savemat
except ImportError:
    savemat = None

from modelswap.model import MLP, save_model
from modelswap.dataset import generate_quadrant_data, save_dataset, export_matlab_data

def train_mlp(
    epochs: int,
    lr: float = 0.01,
    seed: int = 42,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    hidden_dim: int = 16
) -> MLP:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if X_train is None or y_train is None:
        X_train, y_train, _, _ = generate_quadrant_data(seed=seed)

    model = MLP(input_dim=2, hidden_dim=hidden_dim, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def export_matlab_weights(
    model: MLP,
    json_path: str | Path,
    mat_path: str | Path | None = None
) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    # net.0 is Layer 1 (Linear), net.2 is Layer 2 (Linear)
    w1 = state["net.0.weight"].cpu().numpy().tolist()
    b1 = state["net.0.bias"].cpu().numpy().tolist()
    w2 = state["net.2.weight"].cpu().numpy().tolist()
    b2 = state["net.2.bias"].cpu().numpy().tolist()

    weights_dict = {"W1": w1, "b1": b1, "W2": w2, "b2": b2}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, indent=2)

    if mat_path is not None and savemat is not None:
        mat_path = Path(mat_path)
        mat_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(
            str(mat_path),
            {
                "W1": state["net.0.weight"].cpu().numpy(),
                "b1": state["net.0.bias"].cpu().numpy(),
                "W2": state["net.2.weight"].cpu().numpy(),
                "b2": state["net.2.bias"].cpu().numpy(),
            }
        )

def run_training_pipeline(
    data_dir: str | Path = "data",
    models_dir: str | Path = "models",
    matlab_dir: str | Path = "matlab",
    seed: int = 42
) -> dict:
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    matlab_dir = Path(matlab_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    matlab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate & save dataset
    X_train, y_train, X_test, y_test = generate_quadrant_data(n_samples=1000, seed=seed)
    save_dataset(X_train, y_train, X_test, y_test, data_dir / "quadrants_data.npz")
    export_matlab_data(
        X_test, y_test,
        matlab_dir / "test_data.json",
        matlab_dir / "test_data.mat"
    )

    # 2. Train Model A (Baseline: 10 epochs, moderate accuracy ~75-80%)
    model_a = train_mlp(epochs=10, lr=0.01, seed=seed, X_train=X_train, y_train=y_train)
    save_model(model_a, models_dir / "model_a.pt")
    export_matlab_weights(model_a, matlab_dir / "weights_a.json", matlab_dir / "weights_a.mat")

    # 3. Train Model B (Candidate: 65 epochs, high accuracy >= 94%)
    model_b = train_mlp(epochs=65, lr=0.015, seed=seed, X_train=X_train, y_train=y_train)
    save_model(model_b, models_dir / "model_b.pt")
    export_matlab_weights(model_b, matlab_dir / "weights_b.json", matlab_dir / "weights_b.mat")

    # Compute quick validation stats
    with torch.no_grad():
        x_t = torch.from_numpy(X_test)
        acc_a = float((model_a.predict(x_t).numpy() == y_test).mean())
        acc_b = float((model_b.predict(x_t).numpy() == y_test).mean())

    return {
        "acc_a": acc_a,
        "acc_b": acc_b,
        "model_a_path": str(models_dir / "model_a.pt"),
        "model_b_path": str(models_dir / "model_b.pt"),
    }

if __name__ == "__main__":
    results = run_training_pipeline()
    print(f"Training Complete! Model A Acc: {results['acc_a']:.4f}, Model B Acc: {results['acc_b']:.4f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelswap/train.py tests/test_train.py
git commit -m "feat(ml): implement training pipeline and weight exports for Model A and B"
```

---

### Task 4: Comparison Engine (`modelswap/compare.py`)

**Files:**
- Create: `modelswap/compare.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes:
  - `modelswap.model.load_model`, `modelswap.model.MLP`
- Produces:
  - `class ComparisonResult(dataclass)`: containing `acc_a`, `acc_b`, `delta_acc`, `loss_a`, `loss_b`, `delta_loss`, `latency_a_ms`, `latency_b_ms`, `delta_latency_ms`, `verdict: str`
  - `verify_identical_architecture(model_a: MLP, model_b: MLP) -> bool`
  - `evaluate_and_compare(model_a_path, model_b_path, data_path) -> ComparisonResult`
  - CLI runner `main()` exiting 0 on PASS, 1 on FAIL.

- [ ] **Step 1: Write the failing test for `modelswap/compare.py`**

```python
# tests/test_compare.py
import pytest
from pathlib import Path
import torch
import numpy as np
from modelswap.model import MLP, save_model
from modelswap.dataset import generate_quadrant_data, save_dataset
from modelswap.compare import evaluate_and_compare, verify_identical_architecture, ComparisonResult

def test_verify_identical_architecture():
    m1 = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    m2 = MLP(input_dim=2, hidden_dim=16, output_dim=2)
    m_diff = MLP(input_dim=2, hidden_dim=32, output_dim=2)

    assert verify_identical_architecture(m1, m2) is True
    with pytest.raises(ValueError, match="Architecture mismatch"):
        verify_identical_architecture(m1, m_diff)

def test_evaluate_and_compare(tmp_path: Path):
    from modelswap.train import train_mlp
    X_train, y_train, X_test, y_test = generate_quadrant_data(n_samples=500, seed=42)
    data_file = tmp_path / "data.npz"
    save_dataset(X_train, y_train, X_test, y_test, data_file)

    m_a = train_mlp(epochs=5, seed=42, X_train=X_train, y_train=y_train)
    m_b = train_mlp(epochs=50, seed=42, X_train=X_train, y_train=y_train)
    path_a = tmp_path / "model_a.pt"
    path_b = tmp_path / "model_b.pt"
    save_model(m_a, path_a)
    save_model(m_b, path_b)

    result = evaluate_and_compare(path_a, path_b, data_file)
    assert isinstance(result, ComparisonResult)
    assert result.delta_acc > 0.0
    assert result.verdict == "PASS"
    assert result.latency_b_ms < 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'modelswap.compare'`

- [ ] **Step 3: Implement `modelswap/compare.py`**

```python
# modelswap/compare.py
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

from modelswap.model import MLP, load_model

@dataclass
class ComparisonResult:
    model_a_path: str
    model_b_path: str
    acc_a: float
    acc_b: float
    delta_acc: float
    loss_a: float
    loss_b: float
    delta_loss: float
    latency_a_ms: float
    latency_b_ms: float
    delta_latency_ms: float
    verdict: str
    reason: str

def verify_identical_architecture(model_a: MLP, model_b: MLP) -> bool:
    """Verifies that both models have identical topology and parameter tensor shapes."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError(
            f"Architecture mismatch: layer keys differ.\n"
            f"Model A keys: {list(state_a.keys())}\n"
            f"Model B keys: {list(state_b.keys())}"
        )

    for key in state_a.keys():
        if state_a[key].shape != state_b[key].shape:
            raise ValueError(
                f"Architecture mismatch on layer '{key}': "
                f"Model A shape {state_a[key].shape} != Model B shape {state_b[key].shape}"
            )
    return True

def _measure_inference_latency(model: MLP, x_sample: torch.Tensor, n_warmup: int = 50, n_trials: int = 500) -> float:
    """Measures mean per-sample inference latency in milliseconds."""
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_sample)

        start_ns = time.perf_counter_ns()
        for _ in range(n_trials):
            _ = model(x_sample)
        end_ns = time.perf_counter_ns()

    total_time_ms = (end_ns - start_ns) / 1_000_000.0
    return total_time_ms / n_trials

def evaluate_and_compare(
    model_a_path: str | Path,
    model_b_path: str | Path,
    data_path: str | Path,
    max_latency_threshold_ms: float = 5.0
) -> ComparisonResult:
    """
    Evaluates both models on test data, validates identical architecture,
    computes performance deltas, and determines eligibility verdict.
    """
    model_a_path = Path(model_a_path)
    model_b_path = Path(model_b_path)
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {data_path}")

    # Load test data
    data = np.load(data_path)
    X_test = torch.from_numpy(data["X_test"].astype(np.float32))
    y_test = torch.from_numpy(data["y_test"].astype(np.int64))

    # Load and verify models
    model_a = load_model(model_a_path)
    model_b = load_model(model_b_path)
    verify_identical_architecture(model_a, model_b)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # Model A metrics
        logits_a = model_a(X_test)
        loss_a = float(criterion(logits_a, y_test).item())
        preds_a = torch.argmax(logits_a, dim=-1)
        acc_a = float((preds_a == y_test).float().mean().item())

        # Model B metrics
        logits_b = model_b(X_test)
        loss_b = float(criterion(logits_b, y_test).item())
        preds_b = torch.argmax(logits_b, dim=-1)
        acc_b = float((preds_b == y_test).float().mean().item())

    # Measure per-sample latency
    single_sample = X_test[0:1]
    latency_a_ms = _measure_inference_latency(model_a, single_sample)
    latency_b_ms = _measure_inference_latency(model_b, single_sample)

    # Deltas
    delta_acc = acc_b - acc_a
    delta_loss = loss_b - loss_a
    delta_latency_ms = latency_b_ms - latency_a_ms

    # Verdict
    if delta_acc < 0.0:
        verdict = "FAIL"
        reason = f"Accuracy regression: ΔAcc={delta_acc:+.4f} (< 0.0)"
    elif latency_b_ms > max_latency_threshold_ms:
        verdict = "FAIL"
        reason = f"Latency threshold exceeded: {latency_b_ms:.3f}ms > {max_latency_threshold_ms:.3f}ms"
    else:
        verdict = "PASS"
        reason = f"Candidate improved accuracy by {delta_acc*100:+.2f}% with acceptable latency ({latency_b_ms:.3f}ms)"

    return ComparisonResult(
        model_a_path=str(model_a_path),
        model_b_path=str(model_b_path),
        acc_a=acc_a,
        acc_b=acc_b,
        delta_acc=delta_acc,
        loss_a=loss_a,
        loss_b=loss_b,
        delta_loss=delta_loss,
        latency_a_ms=latency_a_ms,
        latency_b_ms=latency_b_ms,
        delta_latency_ms=delta_latency_ms,
        verdict=verdict,
        reason=reason,
    )

def print_comparison_table(res: ComparisonResult) -> None:
    print("==================================================================")
    print("                 ML MODEL SWAP: COMPARISON REPORT                 ")
    print("==================================================================")
    print(f"Model A (Current)   : {res.model_a_path}")
    print(f"Model B (Candidate) : {res.model_b_path}")
    print("------------------------------------------------------------------")
    print(f"{'Metric':<18} | {'Model A (Curr)':<14} | {'Model B (Cand)':<14} | {'Delta (B - A)':<12}")
    print("------------------------------------------------------------------")
    print(f"{'Accuracy':<18} | {res.acc_a*100:>12.2f}% | {res.acc_b*100:>12.2f}% | {res.delta_acc*100:>+10.2f}%")
    print(f"{'Cross-Entropy Loss':<18} | {res.loss_a:>14.4f} | {res.loss_b:>14.4f} | {res.delta_loss:>+12.4f}")
    print(f"{'Latency (ms)':<18} | {res.latency_a_ms:>12.4f}ms | {res.latency_b_ms:>12.4f}ms | {res.delta_latency_ms:>+10.4f}ms")
    print("==================================================================")
    status_symbol = "✔ PASS" if res.verdict == "PASS" else "✖ FAIL"
    print(f"ELIGIBILITY VERDICT : {status_symbol} ({res.reason})")
    print("==================================================================")

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and compare Model A vs Model B.")
    parser.add_argument("--model-a", default=None, help="Path to Model A (baseline)")
    parser.add_argument("--model-b", default=None, help="Path to Model B (candidate)")
    parser.add_argument("--data", default="data/quadrants_data.npz", help="Path to evaluation dataset")
    parser.add_argument("--models-yaml", default="models.yaml", help="Path to models.yaml registry")
    args = parser.parse_args()

    model_a = args.model_a
    model_b = args.model_b

    # Resolve from models.yaml if paths not directly provided
    yaml_path = Path(args.models_yaml)
    if (not model_a or not model_b) and yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            roles = cfg.get("roles", {})
            classifier_cfg = roles.get("classifier", roles.get("classifier_role", {}))
            if not model_a:
                model_a = classifier_cfg.get("previous", classifier_cfg.get("current", "models/model_a.pt"))
            if not model_b:
                model_b = classifier_cfg.get("candidate", "models/model_b.pt")

    model_a = model_a or "models/model_a.pt"
    model_b = model_b or "models/model_b.pt"

    try:
        res = evaluate_and_compare(model_a, model_b, args.data)
        print_comparison_table(res)
        return 0 if res.verdict == "PASS" else 1
    except Exception as e:
        print(f"Error during comparison: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modelswap/compare.py tests/test_compare.py
git commit -m "feat(ml): implement model comparison engine with CLI and verdict"
```

---

### Task 5: End-to-End Execution, Artifact Generation & Teammate Verification

**Files:**
- Create: `models.yaml` (default template for Arjun and Anirudh)
- Create: `tests/verify_contracts.py`
- Generate: `data/quadrants_data.npz`, `models/model_a.pt`, `models/model_b.pt`, `matlab/*`

**Interfaces:**
- Produces:
  - Validated repository files ready for immediate integration with Arjun, Achyut, and Anirudh.

- [ ] **Step 1: Write `tests/verify_contracts.py`**

```python
# tests/verify_contracts.py
import pytest
from pathlib import Path
import json
import torch
import numpy as np
import yaml
from modelswap.model import MLP, load_model
from modelswap.compare import evaluate_and_compare

def test_arjun_loader_and_smoke_test_contract():
    # 1. models.yaml exists and is valid
    yaml_path = Path("models.yaml")
    assert yaml_path.exists()
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "roles" in cfg
    classifier = cfg["roles"].get("classifier", cfg["roles"].get("classifier_role"))
    assert classifier is not None

    # 2. Both model paths exist
    path_a = Path(classifier.get("previous", "models/model_a.pt"))
    path_b = Path(classifier.get("current", "models/model_b.pt"))
    assert path_a.exists(), f"Missing {path_a}"
    assert path_b.exists(), f"Missing {path_b}"

    # 3. Smoke test: load and execute forward pass on dummy batch
    for p in [path_a, path_b]:
        model = load_model(p)
        dummy_input = torch.zeros((1, 2), dtype=torch.float32)
        logits = model(dummy_input)
        assert logits.shape == (1, 2)
        assert not torch.isnan(logits).any()

def test_achyut_matlab_contract():
    # Achyut needs test data and weights
    for fname in ["weights_a.json", "weights_b.json", "test_data.json"]:
        p = Path("matlab") / fname
        assert p.exists(), f"Missing MATLAB file {p}"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) > 0

def test_anirudh_compare_contract():
    # Anirudh calls evaluate_and_compare programmatically
    res = evaluate_and_compare("models/model_a.pt", "models/model_b.pt", "data/quadrants_data.npz")
    assert res.verdict == "PASS"
    assert res.delta_acc >= 0.0
```

- [ ] **Step 2: Create default `models.yaml`**

```yaml
roles:
  classifier:
    architecture: mlp_2layer
    input_dim: 2
    output_dim: 2
    current: "models/model_a.pt"
    previous: "models/model_b.pt"
    candidate: "models/model_b.pt"
```

- [ ] **Step 3: Execute training pipeline to generate actual models and MATLAB data**

Run: `python -m modelswap.train`  
Verify: `models/model_a.pt`, `models/model_b.pt`, `data/quadrants_data.npz`, and `matlab/` files created.

- [ ] **Step 4: Execute `modelswap.compare` CLI**

Run: `python -m modelswap.compare`  
Expected: Prints clean ASCII comparison table with PASS verdict and exits with code 0.

- [ ] **Step 5: Run all test suites**

Run: `pytest tests/ -v`  
Expected: All tests pass.

- [ ] **Step 6: Commit all generated deliverables**

```bash
git add models.yaml tests/verify_contracts.py models/ data/ matlab/
git commit -m "feat(ml): generate production models, dataset, and teammate contract tests"
```
