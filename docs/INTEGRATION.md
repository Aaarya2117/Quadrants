# Integration Guide: the engine behind the CLI

The CLI, the demo and the tests run on a stub engine unless the real engine
module is present. This document describes the contract the CLI expects, and
how the real engine (`modelswap/real_engine.py`) meets it.

The real engine is in place. This guide is for anyone who replaces or extends it.

## 1. The contract

Anything implementing these five methods can drive the CLI and the demo. The
signatures are in `modelswap/engine.py` (`SwapEngine`), and the result types are
in `modelswap/results.py`.

| Method | Returns | Used by |
|---|---|---|
| `status(role)` | `RoleStatus(role, current, previous)` | `status`, `demo` |
| `compare(role, candidate, data=None)` | `CompareResult(model_a, model_b, eligible, reasons, max_latency_ms)` | `compare`, and `apply` (as the gate) |
| `apply(role, candidate)` | `SwapResult(operation="apply", ok, current, previous, smoke_test_passed, error)` | `apply` |
| `rollback(role)` | `SwapResult(operation="rollback", ...)` | `rollback` |
| `reset(role)` | `RoleStatus` | `reset`, `demo` |

Rules every engine must follow:

- **`compare` returns the verdict, it does not compute it.** Set `eligible` and `reasons` from the comparison's own verdict (`modelswap.compare.verdict`). Don't re-derive them.
- **`apply` checks before it writes.** Run the architecture check against the current model and the registry entry, then the smoke test. Write only when both pass. A failed check returns `ok=False` and leaves the registry untouched. It doesn't raise.
- **Raise `EngineError` only when the operation cannot run at all**: no previous model for rollback, unknown role, candidate equal to the active model, candidate file missing.
- **`reset` restores the start state** (Model A current, no previous). It is for rehearsal only.

## 2. Who checks what

| Step | Where it happens | On failure |
|---|---|---|
| Verdict gate (apply only) | `commands.execute`, calls `engine.compare` first | Refused with the reasons, unless `--force` |
| Architecture pre-check | `swap._pointer_change` via `RealEngine.verify` | `ok=False`, `error="pre-check failed: ..."` |
| Smoke test | `swap._pointer_change` via `modelswap.compare.smoke_test` | `ok=False`, `smoke_test_passed=False` |
| Atomic write | `swap.write_registry_atomic` (temp file + `os.replace`) | Only reached when both checks pass |
| Audit record | `swap._audit`, one JSON file per call in `swaps/` | Written for every apply and rollback, pass or fail |

## 3. The real engine

`modelswap/real_engine.py` provides:

```python
def create_engine(registry_path: Path | None = None, audit_dir: Path | None = None) -> RealEngine
```

`RealEngine` takes the registry path, the audit directory, and two optional
hooks: `smoke_test` and `verify`. The hooks default to the real checks; tests
pass fakes.

Backend selection is in `modelswap/engine.py`:

- `--backend auto` (default) uses `RealEngine` when `modelswap/real_engine.py` exists.
- `--backend real` requires it and reports an error if it is missing.
- `--backend stub` always uses the stub.

If `real_engine.py` imports a package that is not installed (torch, for example),
the CLI reports `missing dependency 'torch'`. It does not silently fall back to
the stub.

## 4. Options

| Option | Used by | Meaning |
|---|---|---|
| `--registry PATH` | real engine | Path to `models.yaml` (default `models.yaml`) |
| `--state-file PATH` | stub engine | Path to the stub's JSON state (default `.modelswap/state.json`) |
| `--backend` | all | `auto`, `stub` or `real` |
| `--force` (apply) | verdict gate | Apply even when the verdict is FAIL |

The demo always runs on a temporary copy of the registry and state, so it never
changes the project's `models.yaml` or `swaps/`.

## 5. Module ownership

| Module | Responsibility |
|---|---|
| `modelswap/runtime.py` | Reads `models.yaml`, resolves role aliases, reports status and thresholds |
| `modelswap/swap.py` | Pre-check, smoke test, atomic registry write, rollback, reset, audit |
| `modelswap/compare.py` | Evaluation, architecture signature check, smoke test, verdict (Bhagat's) |
| `modelswap/arch.py` | Checkpoint format and loading (Bhagat's) |
| `modelswap/real_engine.py` | Glues the above into the contract |

## 6. Verification

Run these from the project root after any change to the engine:

1. `python -m pytest`: all tests pass.
2. `python -m modelswap --backend real status`: shows the registry's current model.
3. `python -m modelswap --backend real compare --candidate models/model_b.pt`: shows the real numbers (about 85.1% vs 93.8%) and PASS.
4. `python -m modelswap --backend real apply --candidate <a model that fails the verdict>`: refused, with the reasons printed. Add `--force` only to confirm the override works.
5. `python -m modelswap --backend real demo --fast --show-failure`: ends with `Result: PASS (7/7 steps)`, and the banner says `backend=real`.
6. Confirm `git status` shows no change to `models.yaml` after the demo.

## 7. Checklist for changes to the engine

- [ ] `compare` takes `eligible` and `reasons` from the comparison verdict.
- [ ] `apply` runs the architecture pre-check and the smoke test before writing.
- [ ] A failed check leaves `models.yaml` byte-for-byte unchanged.
- [ ] `rollback` raises `EngineError` when there is no previous model.
- [ ] `reset` restores Model A.
- [ ] Every apply and rollback writes an audit record.
- [ ] The steps in section 6 pass.
