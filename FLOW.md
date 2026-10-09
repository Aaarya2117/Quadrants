# ML Model Swap: Operational Flows (3-Hour Sprint)

Overview of execution flows for evaluating, swapping, rolling back, and simulating two ML models with identical architecture.

---

## Flow 1: Model Comparison
```
[User CLI: compare]
        │
        ▼
Load models.yaml ──► Resolve Model A (Current) & Model B (Candidate)
        │
        ▼
Verify Architecture: Confirm input_dim and output_dim match exactly
        │
        ▼
Run Validation Replay:
  ├── Evaluate Accuracy, Loss, and Latency for Model A
  └── Evaluate Accuracy, Loss, and Latency for Model B
        │
        ▼
Print Paired Comparison Matrix:
  ├── ΔAccuracy = Acc(B) - Acc(A)
  ├── ΔLatency = Latency(B) - Latency(A)
  └── Eligibility Verdict (PASS if ΔAcc ≥ 0 and Latency within threshold)
```

---

## Flow 2: Atomic Swap (Apply)
```
[User CLI: apply]
        │
        ▼
Check Preconditions: Model B exists & passed architecture validation
        │
        ▼
Create Backup: Copy models.yaml -> models.yaml.bak
        │
        ▼
Atomic Write: Update models.yaml
  ├── previous ◄── Model A
  └── current  ◄── Model B
        │
        ▼
Run Smoke Test: Execute single inference forward pass with dummy batch
  ├── PASS ──► Write audit record to swaps/<timestamp>.json ──► Done (Exit 0)
  └── FAIL ──► Restore models.yaml.bak ──► Output error (Exit 1)
```

---

## Flow 3: One-Command Rollback
```
[User CLI: rollback]
        │
        ▼
Read models.yaml ──► Verify "previous" pointer exists
        │
        ▼
Invert Pointers:
  ├── current  ◄── previous (Model A)
  └── previous ◄── current (Model B)
        │
        ▼
Run Smoke Test on Restored Model
  ├── PASS ──► Log rollback to swaps/ ──► Done (Exit 0)
  └── FAIL ──► Alert critical state
```

---

## Flow 4: MATLAB Simulation & Visual Showcase
```
[matlab/simulate_swap.m]
        │
        ├── 1. Load weights W_A and W_B (exported from Python or generated in MATLAB)
        ├── 2. Plot 2D Decision Boundaries (Model A vs Model B side-by-side)
        ├── 3. Plot Weight Delta Heatmap: ΔW = W_B - W_A
        └── 4. Simulate Live Data Stream:
                 • t = 1..50: Inference with Model A
                 • t = 51: SWAP EVENT TRIGGERED
                 • t = 52..100: Seamless inference with Model B
```
