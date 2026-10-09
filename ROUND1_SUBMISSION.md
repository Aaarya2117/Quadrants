# ML Model Swap: Presentation Brief

**Track:** Best Open-Source AI / ML Tool  
**Team:** Arjun, Bhagat, Achyut, Anirudh  

---

## 1. Problem Statement
Deploying retrained or fine-tuned ML models in production is error-prone. Even when two models share the exact same architecture, teams often manually overwrite model paths or redeploy services blindly, without:
1. Side-by-side metric comparison on validation data.
2. Safe, atomic swapping with failure smoke tests and automatic rollback.
3. Instant one-command rollback in $< 10\text{ ms}$.
4. Parameter delta audit ($\Delta W$) verifying that the new weights actually loaded.

---

## 2. Proposed Solution
**`modelswap`** provides an automated, configuration-driven mechanism to benchmark, hot-swap, and roll back ML models behind logical roles.

Application code queries a logical **role** (`get("classifier")`) rather than a hardcoded file path. The engine evaluates candidate models against current models, applies changes atomically via `models.yaml`, verifies output integrity with automated smoke tests, tracks weight delta shifts ($\Delta W$), and generates structured audit logs with zero downtime.

---

## 3. Core Capabilities
1. **Multi-Role Support:** Supports tabular MLP models (`classifier`) and transformer sequence classifiers (`sentiment`: BERT vs RoBERTa).
2. **Side-by-Side Comparison:** Evaluates Accuracy, Loss, and Latency between Current (Model A) and Candidate (Model B) with automated PASS/FAIL guardrails.
3. **Atomic Apply:** Updates registry pointers atomically with automatic rollback on smoke-test failure.
4. **Instant Rollback:** Restores previous model state in under 10ms with one CLI command (`modelswap rollback`).
5. **Interactive CLI Console:** Built-in interactive console for operators to browse models, inspect active roles, and execute swaps safely.
6. **Weight Divergence & Parity:** Calculates Frobenius norms, changed parameter count, and parameter hashes to verify live state.

---

## 4. Live Demo Flow
1. **Launch Console:** Run `python -m modelswap --backend real demo`
2. **Select Model Pair:** Choose `[1] classifier` (MLP) or `[2] sentiment` (BERT vs RoBERTa).
3. **Inspect Status:** Show current baseline model serving inference.
4. **Compare:** Run `compare` to show candidate model improves accuracy (+8.67% for MLP, +1.84% for RoBERTa) within latency budget.
5. **Apply:** Atomically promote candidate with smoke test and weight divergence report.
6. **Rollback:** Instant rollback to confirm resilient recovery.
