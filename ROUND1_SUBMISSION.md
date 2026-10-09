# ML Model Swap: Round 1 Presentation (3-Hour MVP)

**Track:** Best Open-Source AI / ML Tool  
**Team:** Arjun, Bhagat, Achyut, Anirudh  
**Sprint Window:** 3 Hours  

---

## 1. Problem Statement
Deploying retrained or fine-tuned ML models in production is error-prone. Even when two models share the exact same architecture, engineers often manually overwrite model paths or redeploy services blindly, without:
1. Side-by-side metric comparison on validation data.
2. Safe atomic swapping with failure smoke tests.
3. Guaranteed one-command rollback.
4. Visual verification of decision boundary and parameter shifts.

---

## 2. Proposed Solution
**ML Model Swap** provides an automated, configuration-driven mechanism to benchmark, hot-swap, and roll back ML models sharing the same network architecture. 

Application code queries a logical **role** (`get("classifier")`) rather than a hardcoded file path. The engine evaluates candidate models against current models, applies changes atomically via `models.yaml`, verifies output integrity with automated smoke tests, and includes a **MATLAB simulation** to visualize decision boundaries, weight shifts ($\Delta W$), and live data streaming continuity.

---

## 3. Core Features (3-Hour Scope)
1. **Identical-Architecture Model Support:** Supports PyTorch / Scikit-Learn MLP models with matching tensor dimensions.
2. **Side-by-Side Comparison:** Evaluates Accuracy, Loss, and Latency between Current (Model A) and Candidate (Model B).
3. **Atomic Apply:** Updates registry pointers atomically with automatic rollback on smoke-test failure.
4. **Instant Rollback:** Restores previous model state in under 50ms with one CLI command (`modelswap rollback`).
5. **Interactive MATLAB Simulation:** Demonstrates real-time classification changes, decision boundaries, and stream stability before and after the swap event.

---

## 4. Live Demo Flow
1. **Initial State:** Application runs inference with Model A (Baseline, 85.08% test accuracy).
2. **Comparison:** Run `modelswap compare` to show Model B achieves 94% accuracy with identical input/output shapes.
3. **Swap Execution:** Run `modelswap apply` to switch the pointer to Model B in `models.yaml`.
4. **MATLAB Visualizer:** Open `simulate_swap.m` to display:
   - 2D classification boundaries of Model A vs Model B.
   - Live stream timeline illustrating zero downtime during the swap at $t = 50$.
5. **Verification & Rollback:** Run `modelswap rollback` to demonstrate instant return to Model A.
