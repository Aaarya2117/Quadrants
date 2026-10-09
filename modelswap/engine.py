"""Engine contract and backend selection.

Any object with the methods below can drive the CLI and the demo. The stub is
used until the real engine module exists. Once modelswap/real_engine.py defines
create_engine(registry_path, audit_dir), backend "auto" picks it up with no
CLI change.

Integration steps are in docs/INTEGRATION.md.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

from modelswap.results import CompareResult, EngineError, RoleStatus, SwapResult
from modelswap.stub_engine import DEFAULT_STATE_FILE, StubEngine

REAL_MODULE = "modelswap.real_engine"
BACKEND_CHOICES = ("auto", "stub", "real")


class SwapEngine(Protocol):
    def status(self, role: str) -> RoleStatus: ...

    def compare(self, role: str, candidate: str, data: str | None = None) -> CompareResult: ...

    def apply(self, role: str, candidate: str) -> SwapResult: ...

    def rollback(self, role: str) -> SwapResult: ...

    def reset(self, role: str) -> RoleStatus: ...


def load_engine(
    backend: str = "auto",
    state_file: Path | None = None,
    registry: Path | None = None,
    audit_dir: Path | None = None,
) -> tuple[SwapEngine, str]:
    """Return (engine, backend_name). backend_name is "real" or "stub".

    state_file is used by the stub only. registry and audit_dir are used by the
    real engine only.
    """
    if backend not in BACKEND_CHOICES:
        raise EngineError(f"unknown backend '{backend}'")

    if backend in ("auto", "real"):
        try:
            module = importlib.import_module(REAL_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name != REAL_MODULE:
                # The real engine exists but needs a package that is not installed.
                # Report it instead of silently falling back to the stub.
                raise EngineError(f"{REAL_MODULE} failed to import: missing dependency '{exc.name}'") from exc
            if backend == "real":
                raise EngineError(f"--backend real requested but {REAL_MODULE}.py does not exist yet") from exc
        else:
            return module.create_engine(registry, audit_dir), "real"

    return StubEngine(state_file or DEFAULT_STATE_FILE), "stub"
