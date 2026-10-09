# ML Model Swap: Product Requirements Document (PRD)

**Goal:** Build and demonstrate safe, zero-downtime, and reversible swapping of **machine learning models behind logical roles**, verified via pre-flight checks, benchmark comparison, automated smoke testing, and an interactive CLI console.  
**Track:** Best Open-Source AI / ML Tool.

---

## 1. Problem & Core Concept

- **The Problem:** Deploying retrained or improved ML models in production is error-prone. Teams frequently overwrite model weights in place or trigger complex container redeployments without:
  1. Automated pre-flight architecture compatibility checks.
  2. Side-by-side metric comparison and regression guardrails.
  3. Atomic pointer flipping with immediate fallback on crash.
  4. Instant, single-command rollback.
  5. Weight divergence tracking ($\Delta W$) to verify actual parameter updates.
- **The Solution:** `modelswap` — a lightweight ML model swap engine. Applications bind to logical roles (`classifier`, `sentiment`), while the engine evaluates candidates, performs atomic configuration updates, executes smoke tests, tracks weight delta shifts, and enables instant rollback.

---

## 2. Supported Roles & Scope

### In-Scope
1. **Identical-Architecture Role (`classifier`):**
   - 2-layer MLP classifier ($\mathbb{R}^2 \to \mathbb{R}^{16} \to \mathbb{R}^2$).
   - Baseline Model A (85.08% test accuracy) vs Challenger Model B (93.75% test accuracy).
   - Strict architecture signature enforcement (input/output dims, hidden dims, activation).
2. **Transformer NLP Role (`sentiment`):**
   - Fine-tuned `bert-base-uncased` (88.76% accuracy) vs `roberta-base` (90.60% accuracy) on SST-2.
   - Cross-architecture checkpoint resolution with tokenization and contract verification.
3. **Core CLI Commands:**
   - `status`: Show current and rollback pointers.
   - `compare`: Paired accuracy, loss, and p50/p95 latency comparison with PASS/FAIL gate.
   - `apply`: Atomic configuration pointer swap with automated smoke test.
   - `rollback`: Revert to previous model in $< 10\text{ ms}$.
   - `reset`: Restore initial state for rehearsals.
   - `demo`: Interactive console menu and scripted walkthrough.
4. **Weight Divergence Analysis:**
   - Frobenius norm parameter distance $\|W_{\text{new}} - W_{\text{old}}\|_F$.
   - Tensor-by-tensor delta count and SHA256 parameter hashes.

---

## 3. Success Criteria & Service-Level Targets

| Metric | Target | Realized |
|---|---|---|
| Swap Execution Time | $< 50\text{ ms}$ | **$8 - 15\text{ ms}$** |
| Rollback Execution Time | $< 50\text{ ms}$ | **$8 - 12\text{ ms}$** |
| Smoke Test Reliability | 100% on valid models | **100% verified** |
| Regressive Candidate Rejection | 100% caught at compare | **100% verified** |
| Test Coverage | Comprehensive pytest suite | **105 tests passing** |
