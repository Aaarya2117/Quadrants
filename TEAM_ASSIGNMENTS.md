# ML Model Swap: Team Assignments & Sprint Schedule

**Team Members:** Arjun, Bhagat, Achyut, Anirudh  
**Sprint Window:** 3 Hours (180 Minutes)  
**Objective:** End-to-end delivery of the open-source `modelswap` CLI tool and runtime library for safe, zero-downtime model swapping.

---

## 1. Member Ownership & Deliverables

| Member | Focus Area | Core Deliverables |
|---|---|---|
| **Arjun** | **Runtime & Swap Engine** | `models.yaml` registry, `get(role)` dynamic loader, atomic apply (`swap.py`), smoke test runner, and instant rollback. |
| **Bhagat** | **ML Models & Comparison** | Train/export identical-architecture ML models (`model_a.pt` vs `model_b.pt`), fine-tune transformer models (`bert-base-uncased` vs `roberta-base`), and evaluation engine in `compare.py`. |
| **Achyut** | **Weight Divergence & Audit** | Weight divergence engine (`weights.py`), Frobenius norm calculations ($\Delta W$), parameter hashing, and audit log generation in `swaps/`. |
| **Anirudh** | **CLI & Interactive Console** | Interactive CLI interface (`cli.py`), interactive selection menu, walkthrough scripting (`demo.py`), and documentation. |

---

## 2. Execution Timeline

### Phase 1: Foundation
- **Arjun:** Create `models.yaml` structure and write `modelswap/runtime.py` with `get(role)`.
- **Bhagat:** Generate synthetic 2D dataset; train Model A (baseline) and Model B (challenger) using 2-layer MLP architecture.
- **Achyut:** Design weight diffing and Frobenius parameter distance metrics ($\Delta W$).
- **Anirudh:** Initialize repository CLI harness (`status`, `compare`, `apply`, `rollback`).

### Phase 2: Implementation & Safety
- **Arjun:** Implement atomic apply (`models.yaml` update + backup) and `rollback` command with smoke testing.
- **Bhagat:** Wire `compare.py` to evaluate models side-by-side on accuracy, loss, and latency; fine-tune NLP models.
- **Achyut:** Connect weight analysis and SHA256 parameter hashing to `apply` and `rollback` steps.
- **Anirudh:** Connect CLI commands to the runtime and comparison engine.

### Phase 3: Integration & Interactive Demo Console
- **All:** Integrate interactive CLI demo menu (`python -m modelswap --backend real demo`).
- Verify complete cycle:
  1. Show active model (`status`).
  2. Run `compare` (candidate shows higher accuracy and passes gate).
  3. Run `apply` (atomic switch, smoke test passes, $\Delta W$ reported).
  4. Run `rollback` (instant revert to baseline).
  5. Validate 105+ test suite passes in pytest.
