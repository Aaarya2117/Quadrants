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
) -> bool:
    """Run one command and print it. Returns True on success.

    compare returns True whenever it ran, because the eligibility verdict is
    information, not a failure. apply and rollback return the swap's ok flag.
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
        if command in ("apply", "rollback"):
            start = time.perf_counter()
            result = engine.apply(role, candidate) if command == "apply" else engine.rollback(role)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(render_swap(result, elapsed_ms), file=out)
            return result.ok
        raise ValueError(f"unknown command '{command}'")
    except EngineError as exc:
        print(render_error(command, exc), file=out)
        return False
