"""Registry-backed swap, rollback and reset with smoke test, restore-on-failure and audit records."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .results import EngineError, RoleStatus, SwapResult
from .runtime import load_registry as _load, resolve_role as _resolve

INITIAL_CURRENT = "models/model_a.pt"


def write_registry_atomic(path: Path, config: dict) -> None:
    """Write config next to path and os.replace it into place."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _default_smoke_test(model_path: str) -> tuple[bool, str]:
    from .compare import smoke_test

    return smoke_test(model_path)


def _audit(audit_dir: Path, operation: str, role: str, result: SwapResult,
           from_model: str, to_model: str, started: float) -> None:
    now = datetime.now(timezone.utc)
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "operation": operation,
        "role": role,
        "ok": result.ok,
        "from_model": from_model,
        "to_model": to_model,
        "smoke_test_passed": result.smoke_test_passed,
        "error": result.error,
        "timestamp_utc": now.isoformat(),
        "duration_ms": (time.perf_counter() - started) * 1000.0,
    }
    name = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{operation}-{role}.json"
    (audit_dir / name).write_text(json.dumps(record, indent=2) + "\n")


def _pointer_change(path: Path, role: str, operation: str, new_current: str, new_previous: str,
                    old_current: str, old_previous: str | None, config: dict, key: str,
                    smoke_test, audit_dir: Path, started: float) -> SwapResult:
    path = Path(path)
    backup = path.with_name(path.name + ".bak")
    shutil.copy2(path, backup)
    config["roles"][key]["previous"] = new_previous
    config["roles"][key]["current"] = new_current
    write_registry_atomic(path, config)
    check = smoke_test or _default_smoke_test
    try:
        passed, message = check(new_current)
    except Exception as exc:  # a crashing smoke test counts as a failed one
        passed, message = False, f"smoke test raised {type(exc).__name__}: {exc}"
    if passed:
        result = SwapResult(operation, role, True, new_current, new_previous, True)
    else:
        shutil.copy2(backup, path)
        result = SwapResult(operation, role, False, old_current, old_previous, False, message)
    _audit(audit_dir, operation, role, result, old_current, new_current, started)
    return result


def apply_swap(path: Path, role: str, candidate: str, *, smoke_test=None,
               audit_dir: Path = Path("swaps")) -> SwapResult:
    """Point the role at candidate, keeping the old model as previous."""
    config = _load(path)
    key = _resolve(config, role)
    entry = config["roles"][key]
    current = entry.get("current")
    if candidate == current:
        raise EngineError(f"{candidate} is already the current model for {role}")
    if not Path(candidate).is_file():
        raise EngineError(f"candidate model file not found: {candidate}")
    started = time.perf_counter()
    return _pointer_change(path, role, "apply", candidate, current, current,
                           entry.get("previous"), config, key, smoke_test, audit_dir, started)


def rollback_swap(path: Path, role: str, *, smoke_test=None,
                  audit_dir: Path = Path("swaps")) -> SwapResult:
    """Swap current and previous."""
    config = _load(path)
    key = _resolve(config, role)
    entry = config["roles"][key]
    current, previous = entry.get("current"), entry.get("previous")
    if not previous or previous == current:
        raise EngineError(f"no previous model recorded for {role}; nothing to roll back to")
    started = time.perf_counter()
    return _pointer_change(path, role, "rollback", previous, current, current,
                           previous, config, key, smoke_test, audit_dir, started)


def reset_registry(path: Path, role: str) -> RoleStatus:
    """Restore the start state: current and previous both INITIAL_CURRENT."""
    config = _load(path)
    key = _resolve(config, role)
    config["roles"][key]["current"] = INITIAL_CURRENT
    config["roles"][key]["previous"] = INITIAL_CURRENT
    write_registry_atomic(path, config)
    return RoleStatus(role, INITIAL_CURRENT, None)
