# Specification: ML Model Swap - Machine Learning Subsystem

**Date:** 2026-10-09  
**Owner:** Bhagat (ML Models & Comparison Deliverables)  
**Teammate Consumers:** Arjun (Runtime loader / smoke test), Achyut (MATLAB visual simulation), Anirudh (CLI orchestration)  

---

## 1. Executive Summary & Goals
The ML subsystem for ML Model Swap provides:
1. Two trained PyTorch Multi-Layer Perceptron (MLP) binary classifiers sharing an **identical neural architecture**:
   - `model_a.pt` (Baseline / Champion): tuned to modest accuracy (~75–80%).
   - `model_b.pt` (Candidate / Challenger): trained to convergence (~94–98%).
2. A deterministic, reproducible synthetic 2D classification dataset based on 4-quadrant / XOR geometry, saved for Python training and exported for MATLAB simulation.
3. An evaluation and comparison engine (`modelswap/compare.py`) that executes side-by-side benchmarking of accuracy, loss, and latency, returning a structured verdict (`PASS` / `FAIL`) usable both programmatically and via CLI.

All random seeds are strictly fixed (`seed=42`). Every printed or returned metric is dynamically computed at runtime.

---

## 2. Shared Model Architecture & Serialization Contract

### 2.1 Architecture Definition (`modelswap/model.py`)
Both models instantiate the exact same `MLP` class (`torch.nn.Module`):
- **Input dimension:** $d = 2$ features ($x_1, x_2$)
- **Hidden layer:** $h = 16$ units with `ReLU` activation
- **Output dimension:** $c = 2$ classes (raw logits $\mathbb{R}^2$; `softmax` produces class probabilities)
- **Parameters:**
  - $W^{(1)} \in \mathbb{R}^{16 \times 2}, \quad b^{(1)} \in \mathbb{R}^{16}$
  - $W^{(2)} \in \mathbb{R}^{2 \times 16}, \quad b^{(2)} \in \mathbb{R}^{2}$
  - Total trainable parameters: $16 \times 2 + 16 + 2 \times 16 + 2 = 32 + 16 + 32 + 2 = 82$.

### 2.2 Serialization & Teammate Interface
- **For Arjun (`runtime.py`, `get(role)`, smoke tests):**
  - Models are saved as PyTorch state dictionaries via `torch.save(model.state_dict(), path)`.
  - Filepaths: `models/model_a.pt` and `models/model_b.pt`.
  - `modelswap.model.MLP` provides a convenience helper `MLP.load(path)` and standard `model.load_state_dict(torch.load(path))` compatibility.
- **For Achyut (`matlab/simulate_swap.m`):**
  - Model weights are exported to `matlab/weights_a.json`, `matlab/weights_b.json`, and `matlab/weights.mat` (using `scipy.io.savemat` if available, with `.json` guaranteed).
  - Key fields: `W1` (16x2), `b1` (16x1), `W2` (2x16), `b2` (2x1).

---

## 3. Synthetic 2D Dataset (`modelswap/dataset.py`)

### 3.1 Data Generation
- Geometry: 2D 4-quadrant XOR problem:
  $$y = \begin{cases} 1 & \text{if } x_1 \cdot x_2 > 0 \\ 0 & \text{otherwise} \end{cases}$$
  Points are sampled uniformly from $[-2, 2] \times [-2, 2]$ with additive Gaussian noise $\mathcal{N}(0, 0.15^2)$ to create realistic, learnable non-linear decision boundaries.
- Total samples: $N = 1000$ (800 train, 200 test).
- Random seed: strictly set to `seed = 42`.

### 3.2 Dataset Artifacts
- Python training & evaluation: `data/quadrants_data.npz` containing arrays `X_train`, `y_train`, `X_test`, `y_test`.
- MATLAB test stream: `matlab/test_data.json` and `matlab/test_data.mat` containing `X_test` and `y_test` arrays.

---

## 4. Model Training Pipeline (`modelswap/train.py`)

- **Optimizer:** Adam with learning rate $\eta = 0.01$.
- **Loss Function:** `nn.CrossEntropyLoss()`.
- **Model A Training:** Trained for 10 epochs to yield baseline performance:
  - Accuracy: $\approx 75\% - 80\%$
- **Model B Training:** Trained for 60 epochs to achieve converged boundary separation:
  - Accuracy: $\ge 92\%$
- **Output Generation:** Running `python -m modelswap.train` generates:
  - `data/quadrants_data.npz`
  - `models/model_a.pt`
  - `models/model_b.pt`
  - `matlab/weights_a.json`, `matlab/weights_b.json`, `matlab/weights.mat`
  - `matlab/test_data.json`, `matlab/test_data.mat`

---

## 5. Comparison Engine (`modelswap/compare.py`)

### 5.1 Architecture Contract Verification
Before computing metrics, `compare.py` enforces:
1. Input dimension of both models is 2.
2. Output dimension of both models is 2.
3. Every layer name and tensor shape in `state_dict` matches between Model A and Model B exactly.

### 5.2 Computed Metrics
1. **Accuracy:** Fraction of correct predictions over the test set $X_{\text{test}}$.
2. **Loss:** Cross entropy loss over $X_{\text{test}}$.
3. **Latency:** Mean per-sample inference time over 500 warmup and evaluation trials using high-precision `time.perf_counter_ns()`.
4. **$\Delta$ Accuracy:** $\text{Acc}_B - \text{Acc}_A$.
5. **$\Delta$ Loss:** $\text{Loss}_B - \text{Loss}_A$.
6. **$\Delta$ Latency:** $\text{Latency}_B - \text{Latency}_A$.

### 5.3 Verdict Criteria
- Verdict is `PASS` if:
  - $\Delta \text{Acc} \ge 0.0$ (Candidate does not regress on accuracy)
  - $\text{Latency}_B \le 5.0\text{ms}$ (Candidate inference latency meets real-time SLA)
- Otherwise, verdict is `FAIL`.

### 5.4 Programmatic & CLI API
- **Programmatic:**
  ```python
  from modelswap.compare import evaluate_and_compare, ComparisonResult
  result: ComparisonResult = evaluate_and_compare(model_a_path, model_b_path, data_path)
  ```
- **CLI (`python -m modelswap.compare`):**
  Accepts optional `--model-a`, `--model-b`, `--data`, `--models-yaml`.
  Prints an aligned ASCII table of metrics, deltas, and the final eligibility verdict.
  Returns process exit code `0` on `PASS` and `1` on `FAIL`.

---

## 6. Directory Structure
```
Quadrants/
├── data/
│   └── quadrants_data.npz
├── models/
│   ├── model_a.pt
│   └── model_b.pt
├── modelswap/
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── compare.py
├── matlab/
│   ├── weights_a.json
│   ├── weights_b.json
│   ├── weights.mat
│   ├── test_data.json
│   └── test_data.mat
└── tests/
    └── test_ml.py
```

---

## 7. Verification & Testing Strategy
- `tests/test_ml.py` validates:
  1. Identical topology check (shapes, layer count, param count).
  2. Dataset generation integrity (shapes, labels in $\{0, 1\}$, reproducible values).
  3. Model forward pass and output bounds on batch inputs.
  4. Comparison engine evaluation output structure, mathematical delta correctness, and verdict determination.
  5. MATLAB exports validity and schema compliance.
