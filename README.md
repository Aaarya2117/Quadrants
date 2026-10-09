# ML Model Swap

**Zero-downtime, safe, and reversible swapping between machine learning models behind logical roles.**

`modelswap` is an open-source CLI tool and runtime library for safely comparing, hot-swapping, and rolling back production ML models with architectural verification, automated smoke testing, weight divergence reporting ($\Delta W$), and zero downtime.

---

## 🌟 Key Capabilities

- **Role-Based Routing:** Applications resolve models dynamically by logical role (e.g. `get("classifier")`, `get("sentiment")`) rather than hardcoded checkpoint paths.
- **Pre-Flight Architecture Verification:** Validates tensor topologies, input/output dimensions, layer types, and parameter counts before permitting a swap.
- **Side-by-Side Benchmark Replay:** Computes delta accuracy, loss, p50/p95 latency, and sample-level paired disagreement on validation datasets.
- **Atomic Pointer Swapping:** Updates active configuration atomically (`os.replace` semantics) with automatic backup restoration if post-swap validation fails.
- **Automated Smoke Testing:** Verifies contract integrity, shape compliance, and NaN/Inf absence with a dummy batch before committing the swap.
- **Weight Divergence & Parity Audit ($\Delta W$):** Reports Frobenius norms, parameter hashes, and per-tensor weight deltas before and after every swap.
- **Instant 1-Command Rollback:** Reverts to the previous known-good model in $< 10\text{ ms}$.
- **Interactive CLI Demo Console:** Built-in interactive menu to select models and trigger swap operations on demand.

---

## ⚡ Architecture & Roles

`modelswap` supports multiple model families out of the box configured in `models.yaml`:

1. **`classifier` (2-Layer Neural MLP):**
   - **Current Baseline (`models/model_a.pt`):** Trained on subset of synthetic 2D data (85.08% test accuracy).
   - **Candidate Challenger (`models/model_b.pt`):** Fully trained (93.75% test accuracy, $+8.67\%$ net gain).
   - **Topology:** $\text{Linear}(2 \to 16) \to \text{ReLU} \to \text{Linear}(16 \to 2)$.

2. **`sentiment` (Transformer NLP):**
   - **Current Baseline (`models/text/bert_base_uncased.pt`):** Fine-tuned `bert-base-uncased` (88.76% validation accuracy).
   - **Candidate Challenger (`models/text/roberta_base.pt`):** Fine-tuned `roberta-base` (90.60% validation accuracy).

---

## 🚀 Quickstart

### 1. Interactive Demo Console (Recommended)

Launch the interactive console menu to browse models and run commands interactively:

```bash
python -m modelswap --backend real demo
```

You can choose:
- `[1]` Tabular MLP classifier (`model_a.pt` vs `model_b.pt`)
- `[2]` Transformer sentiment classifier (`bert_base_uncased.pt` vs `roberta_base.pt`)
- `[3]` Custom `.pt` model files from workspace
- Action commands: `status`, `compare`, `apply`, `rollback`, `reset`, or full scripted walkthrough

### 2. Automated Demo Walkthrough

Run the non-interactive 6-step end-to-end verification walkthrough:

```bash
python -m modelswap --backend real demo --auto --fast
```

### 3. Individual CLI Commands

#### Check Active Model
```bash
python -m modelswap --backend real status --role classifier
```

#### Compare Baseline vs Candidate
```bash
python -m modelswap --backend real compare --role classifier --candidate models/model_b.pt
```

#### Apply the Atomic Swap
```bash
python -m modelswap --backend real apply --role classifier --candidate models/model_b.pt
```

#### Instantly Roll Back
```bash
python -m modelswap --backend real rollback --role classifier
```

#### Reset to Initial State
```bash
python -m modelswap --backend real reset --role classifier
```

---

## 🛠️ Python Application Integration

Downstream services bind to logical roles without coupling to checkpoint file paths:

```python
from modelswap.runtime import get

# Resolve active model for the "classifier" role
model = get("classifier")

# Execute forward inference
logits = model(features)
```

When an operator executes `apply` or `rollback`, `get("classifier")` transparently serves the new checkpoint on the next request.

---

## 🧪 Testing & Verification

Run the full pytest suite (105+ unit and integration tests):

```bash
pytest
```
