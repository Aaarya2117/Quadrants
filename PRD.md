# ML Model Swap: PRD (3-Hour Sprint MVP)

**Goal:** Build and demonstrate safe, reversible swapping of **two ML models sharing the same architecture**, backed by an interactive MATLAB visual simulation.  
**Time Limit:** 3 Hours.  
**Track:** Best Open-Source AI / ML Tool.

---

## 1. Problem & Core Concept
- **Problem:** Updating ML models in production (e.g., deploying a retrained model checkpoint with the same architecture) is often risky, manual, and lacks instant rollback and visual verification.
- **Solution:** A lightweight swap engine that evaluates Model A vs Model B (same network topology), swaps the active pointer atomically, validates via smoke tests, and visually demonstrates the behavior and parameter changes via MATLAB.

---

## 2. 3-Hour MVP Scope

### In-Scope (Must Build & Show)
1. **Identical Architecture Support:** Two ML models (e.g., PyTorch / Scikit-learn MLP classifiers) with matching input dimension $d$ and output dimension $c$.
2. **Config-Driven Registry (`models.yaml`):** Dynamic resolution of active model (`current` vs `previous`).
3. **CLI Swap Commands:**
   - `compare`: Evaluates Model A vs Model B on validation metrics (accuracy, F1, latency).
   - `apply`: Swaps active pointer, executes smoke test, and logs audit record.
   - `rollback`: Reverts pointer to `previous` in one command.
4. **MATLAB Simulation & Visualizer:**
   - Visual plot comparing decision boundaries of Model A vs Model B.
   - Live simulated stream showing output continuity across the swap event.
   - Heatmap of parameter shift $\Delta W = W_{\text{new}} - W_{\text{old}}$.

### Out-of-Scope (Cut for 3-Hour Sprint)
- Complex heterogeneous model conversions.
- Distributed Kubernetes deployment.
- Web UI dashboard (MATLAB simulation serves as visual showcase).

---

## 3. Success Criteria (Demo Gate)
| Metric | Target |
|---|---|
| Swap Execution Time | $< 50\text{ms}$ (instant pointer switch) |
| Rollback Time | 1 command, $< 50\text{ms}$ |
| Smoke Test | 100% pass on shape and contract integrity |
| Demo Readiness | Working CLI + Live MATLAB figure in $\le 3$ hours |
