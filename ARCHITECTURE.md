# ML Model Swap: Architecture

Minimal, robust architecture designed for safe, zero-downtime swapping and rollback between machine learning models behind logical roles.

---

## 1. System Topology

```
+──────────────────────────────────────────────────────────────+
|                         User / CLI                           |
|       (status | compare | apply | rollback | reset | demo)    |
+──────────────────────────────┬───────────────────────────────+
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   +─────────────────+                  +───────────────────+
   |  Swap Engine    |                  | Weight Diff Engine|
   |  - compare.py   |                  | - weights.py      |
   |  - swap.py      |                  | - Frobenius norm  |
   |  - real_engine  |                  | - Parameter hashes|
   +────────┬────────+                  +───────────────────+
            │                                     │
            ▼                                     ▼
   +───────────────────────────────────────────────────────────+
   |   Registry & Audit Log                                    |
   |   - models.yaml (atomic pointer updates via os.replace)   |
   |   - swaps/<timestamp>_<role>.json (structured audit)      |
   +────────────────────────────┬──────────────────────────────+
                                │
                                ▼
   +───────────────────────────────────────────────────────────+
   |   Application Runtime                                     |
   |   get(role) -> Dynamic Model Loading                      |
   |   - MLP Classifier (models/model_a.pt, model_b.pt)        |
   |   - Sentiment NLP  (bert_base_uncased.pt, roberta_base.pt)|
   +───────────────────────────────────────────────────────────+
```

---

## 2. Model Roles & Architecture Contracts

### 2.1 Tabular MLP Classifier (`classifier`)
- **Topology:** Multi-Layer Perceptron (MLP) Classifier.
- **Layers:** 
  - Input Layer: $\mathbb{R}^2$ ($x_1, x_2$)
  - Hidden Layer: $\mathbb{R}^{16}$ with ReLU activation
  - Output Layer: $\mathbb{R}^2$ (binary classification logits)
- **Signature Check:** `architecture_signature()` extracts metadata directly from checkpoint headers without executing model instantiation.

### 2.2 Transformer Sentiment Classifier (`sentiment`)
- **Topology:** Pretrained transformer sequence classification (`bert-base-uncased` & `roberta-base`).
- **Layers:** 12-layer transformer encoder + sequence classification pooler/head.
- **Contract:** Cross-architecture support with standardized output logit interface and tokenization pipeline.

---

## 3. Core Subsystems

### 3.1 Registry (`models.yaml`)
Stores active pointers and rollback targets:
```yaml
roles:
  classifier:
    kind: mlp
    architecture: mlp_2layer
    input_dim: 2
    output_dim: 2
    current: models/model_a.pt
    previous: null
  sentiment:
    kind: text_classification
    cross_architecture: true
    current: models/text/bert_base_uncased.pt
    previous: null
```

### 3.2 Dynamic Runtime Loader (`modelswap/runtime.py`)
```python
def get(role: str):
    config = load_registry("models.yaml")
    active_path = config["roles"][role]["current"]
    return load_model(active_path)
```

### 3.3 Atomic Apply & Rollback Engine (`modelswap/swap.py`, `modelswap/real_engine.py`)
- Writes updated YAML via atomic file replacement (`os.replace`).
- Runs forward-pass smoke test on synthetic input batch.
- Automatically reverts `models.yaml.bak` on failure.
- Appends JSON record to `swaps/`.

### 3.4 Weight Divergence & Verification (`modelswap/weights.py`)
- Analyzes parameter tensors before and after swap.
- Computes Frobenius norm of parameter shift:
  $$\Delta W = \| W_{\text{new}} - W_{\text{old}} \|_F$$
- Reports SHA256 parameter hash verification to guarantee new weights are active.
