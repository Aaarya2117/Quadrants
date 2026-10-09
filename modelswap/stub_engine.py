"""Stub engine: deterministic numbers and a JSON state file.

Lets the CLI and the demo run end to end before the real runtime and compare.py
exist. The metrics are illustrative (they match the 82% -> 94% story in
ROUND1_SUBMISSION.md); they are not computed from any model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modelswap.results import (
    DEFAULT_MAX_LATENCY_MS,
    CompareResult,
    EngineError,
    ModelMetrics,
    RoleStatus,
    SwapResult,
)

DEFAULT_STATE_FILE = Path(".modelswap") / "state.json"
INITIAL_CURRENT = "models/model_a.pt"

# A candidate whose file name contains this marker fails the smoke test.
# The demo uses it for the "rejected swap" fallback.
BROKEN_MARKER = "broken"

# File stem -> (accuracy, latency_ms). Illustrative values only.
STUB_METRICS = {"model_a": (0.82, 1.9), "model_b": (0.94, 2.1)}
UNKNOWN_METRICS = (0.80, 2.5)


class StubEngine:
    """Implements the SwapEngine contract (see modelswap.engine) with a local JSON file."""

    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = Path(state_file)

    # -- state -----------------------------------------------------------------

    def _load(self) -> dict:
        if not self.state_file.exists():
            return {}
        return json.loads(self.state_file.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_file)  # atomic replace, as in the real engine

    @staticmethod
    def _record(data: dict, role: str) -> dict:
        return data.setdefault(role, {"current": INITIAL_CURRENT, "previous": None})

    # -- contract --------------------------------------------------------------

    def status(self, role: str) -> RoleStatus:
        rec = self._record(self._load(), role)
        return RoleStatus(role, rec["current"], rec["previous"])

    def compare(self, role: str, candidate: str, data: str | None = None) -> CompareResult:
        current = self.status(role).current
        return CompareResult(
            role=role,
            model_a=_metrics(current),
            model_b=_metrics(candidate),
            max_latency_ms=DEFAULT_MAX_LATENCY_MS,
        )

    def apply(self, role: str, candidate: str) -> SwapResult:
        data = self._load()
        rec = self._record(data, role)
        if candidate == rec["current"]:
            raise EngineError(f"{candidate} is already the active model for role '{role}'")
        if BROKEN_MARKER in candidate:
            # Same outcome as the real engine: nothing is written, the active model stays.
            return SwapResult(
                operation="apply",
                role=role,
                ok=False,
                current=rec["current"],
                previous=rec["previous"],
                smoke_test_passed=False,
                error="smoke test failed: forward pass returned the wrong output shape (stub)",
            )
        rec["previous"], rec["current"] = rec["current"], candidate
        self._save(data)
        return SwapResult("apply", role, True, rec["current"], rec["previous"], True)

    def rollback(self, role: str) -> SwapResult:
        data = self._load()
        rec = self._record(data, role)
        if rec["previous"] is None:
            raise EngineError(f"no previous model recorded for role '{role}'; nothing to roll back to")
        rec["current"], rec["previous"] = rec["previous"], rec["current"]
        self._save(data)
        return SwapResult("rollback", role, True, rec["current"], rec["previous"], True)

    def reset(self, role: str) -> RoleStatus:
        data = self._load()
        data[role] = {"current": INITIAL_CURRENT, "previous": None}
        self._save(data)
        return self.status(role)


def _metrics(model_path: str) -> ModelMetrics:
    stem = Path(model_path).stem
    accuracy, latency_ms = STUB_METRICS.get(stem, UNKNOWN_METRICS)
    return ModelMetrics(model=model_path, accuracy=accuracy, latency_ms=latency_ms)
