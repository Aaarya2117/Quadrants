# ML Model Swap: 3-Hour Team Assignments & Sprint Schedule

**Team Members:** Arjun, Bhagat, Achyut, Anirudh  
**Sprint Window:** 3 Hours (180 Minutes)  
**Objective:** Live demo of swapping two ML models with identical architecture + interactive MATLAB simulation.

---

## 1. Member Ownership & Deliverables

| Member | Focus Area | Core 3-Hour Deliverables |
|---|---|---|
| **Arjun** | **Runtime & Swap Engine** | `models.yaml` registry, `get(role)` dynamic loader, atomic apply (`swap.py`), smoke test runner, and instant rollback. |
| **Bhagat** | **ML Models & Comparison** | Train/export two identical-architecture ML models (`model_a.pt` vs `model_b.pt`), synthetic validation dataset, and evaluation metrics in `compare.py`. |
| **Achyut** | **MATLAB Simulation** | Interactive `matlab/simulate_swap.m`: 2D decision boundary visualization, weight delta matrix heatmap ($\Delta W$), and live data stream simulation. |
| **Anirudh** | **CLI & Demo Orchestration** | CLI interface (`cli.py`), end-to-end demo workflow scripting, slide/pitch deck, and screen recording backup. |

---

## 2. 3-Hour Execution Timeline

### Hour 1: Foundation (0:00 – 1:00)
- **Arjun:** Create `models.yaml` structure and write `modelswap/runtime.py` with `get(role)`.
- **Bhagat:** Generate synthetic 2D classification dataset; train Model A (baseline) and Model B (improved) using the same 2-layer MLP architecture.
- **Achyut:** Set up MATLAB workspace and write the decision boundary plotting function.
- **Anirudh:** Initialize repository CLI harness (`modelswap compare/apply/rollback`) and draft 3-minute pitch outline.

### Hour 2: Implementation & Visualization (1:00 – 2:00)
- **Arjun:** Implement atomic apply (`models.yaml` update + backup) and `rollback` command with smoke testing.
- **Bhagat:** Wire `compare.py` to evaluate both models side-by-side on accuracy and latency.
- **Achyut:** Finalize `simulate_swap.m` with live streaming plot showing inference before and after the swap event.
- **Anirudh:** Connect CLI commands to the runtime and comparison engine.

### Hour 3: Integration & Demo Rehearsal (2:00 – 3:00)
- **All:** Run complete end-to-end cycle:
  1. Show app running with Model A.
  2. Run `modelswap compare` (Model B shows higher accuracy).
  3. Run `modelswap apply` (atomic switch to Model B, smoke test passes).
  4. Run MATLAB visualizer demonstrating real-time shift in decision boundary and stream stability.
  5. Run `modelswap rollback` (verifying instant revert to Model A).
- **Anirudh & Arjun:** Rehearse live presentation and record backup video.
