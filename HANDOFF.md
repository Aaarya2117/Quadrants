# ML Model Swap: Teammate Handoff & Contract Specification

**Date:** 2026-10-09  
**Deliverables Owner:** Bhagat (ML Subsystem)  
**Teammate Consumers:** Arjun (Runtime & Swap Engine), Achyut (MATLAB Simulation), Anirudh (CLI Orchestration)  

---

## 1. For Arjun (Runtime Loader & Swap Engine)

### Checkpoint File Paths
- **Current / Baseline:** `models/model_a.pt`
- **Candidate / Challenger:** `models/model_b.pt`
- **Registry:** `models.yaml` (default role `classifier` points to `models/model_a.pt` and candidate `models/model_b.pt`)

### Checkpoint Format & Inspection
Each `.pt` file is a Python dictionary containing:
```python
{
    "architecture": "mlp_2layer",
    "input_dim": 2,
    "hidden_dim": 16,
    "output_dim": 2,
    "activation": "relu",
    "state_dict": {
        "fc1.weight": torch.Size([16, 2]),
        "fc1.bias":   torch.Size([16]),
        "fc2.weight": torch.Size([2, 16]),
        "fc2.bias":   torch.Size([2]),
    }
}
```

### Inspecting Signatures (without model build)
```python
from modelswap.arch import architecture_signature

sig = architecture_signature("models/model_a.pt")
# returns: {"architecture": "mlp_2layer", "input_dim": 2, "hidden_dim": 16, "output_dim": 2, "activation": "relu"}
```

### Loading the Model
```python
from modelswap.arch import load_checkpoint

model = load_checkpoint("models/model_a.pt")
# Returns ready-to-evaluate MLP instance in eval() mode
```

### Smoke Test Function (for `apply` and `rollback`)
Import this exact function in your swap engine:
```python
from modelswap.compare import smoke_test

ok, message = smoke_test("models/model_b.pt")
if not ok:
    # Trigger instant rollback!
    print(f"Smoke test failed: {message}")
```
- **Validation performed:** Checkpoint integrity, forward pass on dummy batch, shape check `(1, 2)`, finite value check (no NaN/Inf), and softmax normalization check.

---

## 2. For Achyut (MATLAB Visual Simulation)

### Export Files & Paths
Generated in `exports/` via `python tools/export_for_matlab.py`:
- `exports/weights_a.mat` (Model A baseline weights)
- `exports/weights_b.mat` (Model B candidate weights)
- `exports/data.mat` (1,200 test samples and pre-computed predictions)

### MATLAB Variable Names & Matrix Shapes
All weight matrices are exported **exactly as PyTorch stores them** (no transposes):
| File | Variable | Shape | Description |
|---|---|---|---|
| `weights_a.mat` / `weights_b.mat` | `W1` | `16 x 2` ($h \times d$) | First layer weight matrix |
| `weights_a.mat` / `weights_b.mat` | `b1` | `16 x 1` ($h \times 1$) | First layer bias vector |
| `weights_a.mat` / `weights_b.mat` | `W2` | `2 x 16` ($c \times h$) | Second layer weight matrix |
| `weights_a.mat` / `weights_b.mat` | `b2` | `2 x 1` ($c \times 1$) | Second layer bias vector |
| `data.mat` | `X` | `1200 x 2` ($n \times d$) | 2D test feature matrix |
| `data.mat` | `y` | `1200 x 1` ($n \times 1$) | Ground-truth labels $\{0, 1\}$ |
| `data.mat` | `pred_a` | `1200 x 1` | Python-computed Model A predictions |
| `data.mat` | `pred_b` | `1200 x 1` | Python-computed Model B predictions |

### Standardization Notice
> **CRITICAL:** Feature matrix `X` in `exports/data.mat` and `data/test.csv` is **ALREADY STANDARDIZED** using training set statistics ($\mu, \sigma$).
> **Do NOT re-standardize $X$ in MATLAB `simulate_swap.m`.**

### MATLAB Forward Pass Formulation
In MATLAB, evaluate forward pass using column vector logic:
```matlab
% For sample column vector x (2x1):
h1 = max(0, W1 * x + b1);   % (16x1) ReLU activation
logits = W2 * h1 + b2;      % (2x1) Logits
[~, pred] = max(logits);    % 1-based index (1 or 2)
pred = pred - 1;            % Convert to 0-based label (0 or 1)
```

---

## 3. For Anirudh (CLI Orchestration)

### Calling the Comparison Engine Programmatically
```python
from modelswap.compare import compare, print_comparison, write_json

# Call via role name (reads models.yaml)
result = compare("classifier", data_path="data/test.npz")

# Or call via explicit paths
result = compare(("models/model_a.pt", "models/model_b.pt"), data_path="data/test.npz")

# Print formatted ASCII table
print_comparison(result)

# Save report JSON
write_json(result, "exports/compare_results.json")
```

### CLI Command Invocation
```bash
# Using models.yaml role
python -m modelswap.compare --role classifier

# Explicit arguments with JSON report export
python -m modelswap.compare --model-a models/model_a.pt --model-b models/model_b.pt --data data/test.npz --json-out exports/report.json
```
- **Exit code:** Returns `0` on `PASS` and `1` on `FAIL`.

### Output JSON Shape (`result` dict)
```json
{
  "current_path": "models/model_a.pt",
  "candidate_path": "models/model_b.pt",
  "architecture_ok": true,
  "architecture_mismatches": [],
  "current": {
    "accuracy": 0.8508,
    "loss": 0.3330,
    "latency_p50_ms": 0.0138,
    "latency_p95_ms": 0.0188,
    "throughput_samples_per_sec": 32885722.0
  },
  "candidate": {
    "accuracy": 0.9375,
    "loss": 0.1783,
    "latency_p50_ms": 0.0137,
    "latency_p95_ms": 0.0144,
    "throughput_samples_per_sec": 26619343.0
  },
  "deltas": {
    "accuracy": 0.0867,
    "loss": -0.1547,
    "latency_p50_ms": -0.0001,
    "latency_p95_ms": -0.0044
  },
  "paired_agreement": {
    "differ_count": 130,
    "candidate_correct_current_wrong": 117,
    "current_correct_candidate_wrong": 13,
    "net_gain_samples": 104
  },
  "verdict": "PASS",
  "verdict_reasons": [ ... ]
}
```

---

## 4. Real Computed Benchmark Numbers for the Live Demo

*All numbers dynamically computed from real evaluation on the 1,200-sample test set (never hardcoded):*

| Metric | Model A (Current Baseline) | Model B (Candidate Challenger) | Delta ($\Delta$) |
|---|---|---|---|
| **Test Accuracy** | **85.08%** (1,021 / 1,200) | **93.75%** (1,125 / 1,200) | **+8.67%** (+104 samples) |
| **Cross-Entropy Loss** | **0.3330** | **0.1783** | **-0.1547** |
| **p50 Latency** | **0.0138 ms** | **0.0137 ms** | **-0.0001 ms** |
| **p95 Latency** | **0.0188 ms** | **0.0144 ms** | **-0.0044 ms** |
| **Decision Disagreements**| — | — | **130 samples** (B correct: 117, A correct: 13) |
| **Swap Verdict** | — | — | **PASS** (zero regressions, SLA met) |
