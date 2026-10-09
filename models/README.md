# Models: Origin & Training Provenance

This directory contains the serialized PyTorch model checkpoints for the **ML Model Swap** sprint:
- `models/model_a.pt` (Baseline / Current)
- `models/model_b.pt` (Candidate / Challenger)

---

## Provenance & Training Methodology

To maintain absolute scientific honesty and engineering transparency:
- **Model A (Baseline):** Trained from scratch using `models/train_a.py` on a small subset of the training split (**250 samples** out of 3,600) for a limited duration (**15 epochs**, learning rate $\eta = 0.015$, seed $42$). This yields a deliberately modest baseline accuracy of **85.08%** on the 1,200-sample test set.
- **Model B (Candidate):** Produced by loading Model A's checkpoint and continuing fine-tuning via `models/train_b.py` on the **FULL training split** (**3,600 samples**) for **100 epochs** at a lower learning rate ($\eta = 0.008$, seed $42$). This brings the converged test accuracy to **93.75%** (+8.67 percentage points).

**Neither model uses external data or altered network geometry.** Both models share an identical 2-layer MLP architecture ($2 \to 16 \to 2$ with ReLU activation). Model B represents a classic production scenario: an existing model fine-tuned on more representative data for longer convergence.

---

## Verification & Reproducibility

Every run is strictly deterministic (`seed=42`). Running:
```bash
python data/make_data.py
python models/train_a.py
python models/train_b.py
```
reproduces the exact same test accuracies (**85.08%** vs **93.75%**).
