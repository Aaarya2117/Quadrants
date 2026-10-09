"""Runs one engine operation and prints the result.

Both the CLI and the demo go through execute(), so the demo shows exactly what
the CLI does.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

from modelswap.engine import SwapEngine
from modelswap.render import render_compare, render_error, render_status, render_swap
from modelswap.results import EngineError

COMMANDS = ("status", "compare", "apply", "rollback", "reset")


def execute(
    engine: SwapEngine,
    command: str,
    role: str,
    candidate: str | None = None,
    data: str | None = None,
    out: TextIO | None = None,
    force: bool = False,
) -> bool:
    """Run one command and print it. Returns True on success.

    compare returns True whenever it ran, because the verdict is information,
    not a failure. apply refuses a candidate whose verdict is not PASS unless
    force is True. apply and rollback return the swap's ok flag.
    """
    out = out or sys.stdout
    try:
        if command == "status":
            print(render_status(engine.status(role)), file=out)
            return True
        if command == "reset":
            print(render_status(engine.reset(role)), file=out)
            return True
        if command == "compare":
            print(render_compare(engine.compare(role, candidate, data)), file=out)
            return True
        if command == "apply":
            if not force:
                report = engine.compare(role, candidate, data)
                if not report.eligible:
                    print(render_compare(report), file=out)
                    print(
                        f"[FAIL] apply refused: candidate did not pass the verdict "
                        f"(use --force to override)",
                        file=out,
                    )
                    return False
            start = time.perf_counter()
            result = engine.apply(role, candidate)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(render_swap(result, elapsed_ms), file=out)
            return result.ok
        if command == "rollback":
            start = time.perf_counter()
            result = engine.rollback(role)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(render_swap(result, elapsed_ms), file=out)
            return result.ok
        raise ValueError(f"unknown command '{command}'")
    except ModuleNotFoundError as exc:
        # Missing optional dependency (e.g. torch): report it cleanly instead of a traceback.
        print(render_error(command, f"needs a package that is not installed: '{exc.name}'"), file=out)
        return False
    except EngineError as exc:
        print(render_error(command, exc), file=out)
        return False
