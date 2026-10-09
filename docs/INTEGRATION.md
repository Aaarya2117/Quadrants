# Integrating the Real Engine

The CLI, the demo and the tests run on a stub until the real engine exists. This
document describes how Arjun's runtime/swap code and Bhagat's comparison code
plug into the `modelswap` package with no changes to the CLI or demo.

## 1. The contract

Anything implementing these five methods can drive the CLI and demo. The
signatures are in `modelswap/engine.py` (`SwapEngine`), and the return types are in
`modelswap/results.py`.

| Method | Returns | Used by |
|---|---|---|
| `status(role)` | `RoleStatus(role, current, previous)` | `status`, `demo` |
| `compare(role, candidate, data=None)` | `CompareResult(model_a, model_b, max_latency_ms)` | `compare` |
| `apply(role, candidate)` | `SwapResult(operation="apply", ok, current, previous, smoke_test_passed, error)` | `apply` |
| `rollback(role)` | `SwapResult(operation="rollback", ...)` | `rollback` |
| `reset(role)` | `RoleStatus` | `reset`, `demo` |

Rules the real engine must follow:
- `apply` with a failing smoke test returns `ok=False`, restores `models.yaml` from the backup, and leaves `current` unchanged. It does not raise.
- Raise `EngineError` only when the operation cannot run at all (no previous model for rollback, unknown role, candidate equal to the active model).
- `compare` returns raw numbers. The PASS/FAIL verdict comes from `CompareResult.eligible`, so the policy stays in one place.
- `reset` restores the initial registry state (Model A as current, no previous). It exists for rehearsal only.

## 2. The one file to add

Create `modelswap/real_engine.py` with a factory function:

```python
from pathlib import Path

from modelswap.results import CompareResult, RoleStatus, SwapResult


class RealEngine:
    def __init__(self, registry_path: Path):
        ...  # load models.yaml, set up backups and swaps/ audit dir

    def status(self, role: str) -> RoleStatus: ...
    def compare(self, role: str, candidate: str, data: str | None = None) -> CompareResult: ...
    def apply(self, role: str, candidate: str) -> SwapResult: ...
    def rollback(self, role: str) -> SwapResult: ...
    def reset(self, role: str) -> RoleStatus: ...


def create_engine(state_file: Path | None) -> RealEngine:
    # state_file is the --state-file option; use it for models.yaml, or
    # return RealEngine(Path("models.yaml")) if the registry path is fixed.
    return RealEngine(state_file or Path("models.yaml"))
```

Then `load_engine` picks it up automatically:

- `--backend auto` (default) uses `RealEngine` once the file exists.
- `--backend real` requires it and reports an error if it is missing.
- `--backend stub` always uses the stub.

If `real_engine.py` imports a package that is not installed (torch, for example), the CLI reports `missing dependency 'torch'`. It does not silently fall back to the stub.

## 3. Mapping the team's modules

| Real module (owner) | Used by `RealEngine` for |
|---|---|
| `modelswap/runtime.py` (Arjun): `load_registry`, `get(role)` | `status`, and loading models for `compare` |
| `modelswap/swap.py` (Arjun): atomic apply, backup, rollback, smoke test | `apply`, `rollback`, `reset` |
| `compare.py` (Bhagat): accuracy, F1, latency | `compare` |

Arjun and Bhagat keep their own file layout. `RealEngine` is the only place
that knows about them.

## 4. Verification after integration

1. `python -m unittest discover -s tests -t .` still passes. The tests cover the stub and the backend selection. Add a contract test for `RealEngine` (see step 5).
2. `python -m modelswap --backend real reset` runs without error.
3. `python -m modelswap --backend real compare --candidate models/model_b.pt` shows real metrics, not 82% / 94%.
4. `python -m modelswap --backend real demo --fast --show-failure` ends with `Result: PASS (7/7 steps)` and the banner says `backend=real`.
5. Add a contract test: run the same operations against `RealEngine` (using a temporary registry and tiny test models) and check the result types and the `ok`/`error` rules listed above.

## 5. Checklist for the integrator

- [ ] `modelswap/real_engine.py` defines `create_engine(state_file)` and a class that satisfies `SwapEngine`.
- [ ] `apply` returns `ok=False` on smoke-test failure and leaves `models.yaml` unchanged.
- [ ] `rollback` raises `EngineError` when there is no previous model.
- [ ] `compare` returns real numbers; `eligible` is not reimplemented.
- [ ] `reset` restores Model A.
- [ ] `docs/DEMO_RUNBOOK.md` commands work against the real engine.
- [ ] The banner in `demo` shows `backend=real`.
