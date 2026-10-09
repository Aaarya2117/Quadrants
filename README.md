# ML Model Swap (3-Hour Sprint)

**Instant, safe, and reversible swapping between two ML models with identical architecture.**

Compare two machine learning models sharing the exact same architecture (e.g., Model A vs Model B / Champion vs Challenger), verify accuracy and drift, hot-swap the active model via configuration, and visualize the live transition in MATLAB. Roll back in one command.

---

## 🎯 3-Hour Hackathon Demo Scope

- **Target:** Two ML models with identical architecture (e.g., 2-layer MLP / Neural Classifier with matching input $\mathbb{R}^d$ and output $\mathbb{R}^c$).
- **Core Action:** Benchmark Model A vs Model B on test dataset $\to$ Atomic swap in `models.yaml` $\to$ Automated smoke test $\to$ 1-command rollback.
- **Visual Demo:** Interactive **MATLAB Simulation** displaying decision boundary shift, weight divergence ($\Delta W$), and real-time swap transition.

---

## ⚡ How It Works

### 1. Application calls a role, not hardcoded files
```python
from modelswap.runtime import get

# App queries the active model dynamically
model = get("classifier")
prediction = model.predict(features)
```

### 2. Registry (`models.yaml`)
```yaml
roles:
  classifier:
    architecture: mlp_classifier_v1
    input_dim: 10
    output_dim: 2
    current: models/model_b.pt      # Candidate (newly swapped)
    previous: models/model_a.pt     # Rollback target
    metric_threshold:
      min_accuracy: 0.90
      max_latency_ms: 15.0
```

---

## 🚀 3-Minute Quickstart

### Step 1: Run the Comparison
```bash
python -m modelswap compare --role classifier --candidate models/model_b.pt
```
Compares Model A vs Model B on accuracy, loss, and latency.

### Step 2: Apply the Swap
```bash
python -m modelswap apply --role classifier --candidate models/model_b.pt
```
Atomically updates `models.yaml`, runs a smoke test, and creates an audit record.

### Step 3: Instant Rollback
```bash
python -m modelswap rollback --role classifier
```
Restores Model A immediately if issues occur.

---

## 📊 MATLAB Simulation & Visualization

We include a MATLAB script (`matlab/simulate_swap.m`) to visually demonstrate the swap:
1. **Decision Boundary Plot:** Shows classification boundary of Model A vs Model B.
2. **Weight Delta Heatmap:** Displays parameter divergence $\Delta W = W_B - W_A$.
3. **Live Swap Monitor:** Plots inference stream before, during, and after the swap trigger with zero system downtime.

To run:
```matlab
% In MATLAB command window or Octave:
run('matlab/simulate_swap.m')
```
