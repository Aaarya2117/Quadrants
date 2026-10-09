# ML Model Swap: Architecture (3-Hour Sprint)

Minimal, robust architecture designed to demonstrate safe swapping and rollback between two machine learning models with identical architecture in under 3 hours.

---

## 1. System Topology

```
+──────────────────────────────────────────────────────────────+
|                         User / CLI                           |
|       (modelswap compare | modelswap apply | rollback)       |
+──────────────────────────────┬───────────────────────────────+
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   +─────────────────+                  +───────────────────+
   |  Swap Engine    |                  | MATLAB Simulator  |
   |  - compare.py   |                  | simulate_swap.m   |
   |  - swap.py      |                  | - Decision bounds |
   |  - rollback.py  |                  | - Weight delta ΔW |
   +────────┬────────+                  | - Stream monitor  |
            │                           +───────────────────+
            ▼
   +──────────────────────────────────+
   |   Registry & Config              |
   |   models.yaml (current/previous) |
   +────────────────┬─────────────────+
                    │
                    ▼
   +──────────────────────────────────+
   |   Application Runtime            |
   |   get(role) -> ML Model A or B   |
   +──────────────────────────────────+
```

---

## 2. Same-Architecture Model Contract

Both Model A (Current/Champion) and Model B (Candidate/Challenger) share identical topology:
- **Architecture Type:** Multi-Layer Perceptron (MLP) Classifier / Neural Network.
- **Layers:** 
  - Input Layer: $\mathbb{R}^d$ ($d$ features)
  - Hidden Layer: $\mathbb{R}^h$ with activation function (e.g., ReLU / Tanh)
  - Output Layer: $\mathbb{R}^c$ ($c$ classes or probability logits)
- **Weight Matrices:**
  - $W^{(1)} \in \mathbb{R}^{d \times h}, \quad b^{(1)} \in \mathbb{R}^h$
  - $W^{(2)} \in \mathbb{R}^{h \times c}, \quad b^{(2)} \in \mathbb{R}^c$

Because dimensions match, switching models involves **zero code or interface changes** in downstream client code.

---

## 3. Core Components

### 3.1 Registry (`models.yaml`)
Stores the dynamic pointer to the active model file:
```yaml
roles:
  classifier_role:
    architecture: mlp_2layer
    input_dim: 2
    output_dim: 2
    current: "models/model_b.pt"
    previous: "models/model_a.pt"
```

### 3.2 Dynamic Runtime Loader (`modelswap/runtime.py`)
```python
def get(role: str):
    config = load_registry("models.yaml")
    active_path = config["roles"][role]["current"]
    return load_model(active_path)
```

### 3.3 Atomic Apply & Rollback Engine (`modelswap/swap.py`)
- Writes updated YAML via atomic file replacement (`os.replace`).
- Runs forward-pass smoke test on synthetic input batch.
- Automatically reverts on failure.
- Appends JSON record to `swaps/`.

### 3.4 MATLAB Visual Simulation (`matlab/simulate_swap.m`)
- Imports or generates weights $W_A$ and $W_B$.
- Plots 2D decision boundary showing classification performance difference.
- Computes weight shift:
  $$\Delta W = \| W_B - W_A \|_F$$
- Simulates a real-time data stream displaying inference predictions across the swap boundary.
