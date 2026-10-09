# ML Model Swap: Operational Flows

Overview of execution flows for evaluating, swapping, rolling back, and auditing ML models behind logical roles.

---

## Flow 1: Model Comparison
```
[User CLI: compare]
        │
        ▼
Load models.yaml ──► Resolve Model A (Current) & Model B (Candidate)
        │
        ▼
Verify Architecture: Confirm input_dim, output_dim, and layer types match
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
  ├── Paired Disagreement Analysis
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
Atomic Write: Update models.yaml (os.replace)
  ├── previous ◄── Model A
  └── current  ◄── Model B
        │
        ▼
Run Smoke Test: Execute single inference forward pass with dummy batch
        │
  ├── PASS ──► Compute Weight Divergence (ΔW) ──► Log to swaps/<timestamp>.json ──► Done (Exit 0)
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
  ├── PASS ──► Log rollback to swaps/ ──► Report weight revert ──► Done (Exit 0)
  └── FAIL ──► Alert critical state
```

---

## Flow 4: Interactive Demo Console
```
[User CLI: python -m modelswap --backend real demo]
        │
        ▼
1. Detect Models & Read Presets:
     [1] classifier (MLP Model A vs Model B)
     [2] sentiment  (BERT vs RoBERTa)
     [3] Custom .pt files
        │
        ▼
2. User Selects Model Pair
        │
        ▼
3. Interactive Command Loop:
     [1] status   ──► Query active role and previous rollback pointer
     [2] compare  ──► Run validation replay and print delta table
     [3] apply    ──► Atomically swap active pointer with smoke test
     [4] rollback ──► Instantly restore previous model
     [5] reset    ──► Return to baseline state
     [6] demo     ──► Run scripted 6-step walkthrough
     [7] switch   ──► Switch to another model pair
     [0] exit     ──► Exit console
```
