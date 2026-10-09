"""Scripted live demo: the README flow, one step at a time.

Each step runs through commands.execute(), the same path the CLI uses. The demo
pauses for Enter between steps unless fast=True (rehearsal).

    python -m modelswap demo                 # live, waits for Enter
    python -m modelswap demo --fast          # no pauses
    python -m modelswap demo --show-failure  # also shows a rejected swap
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, TextIO

from modelswap.commands import execute
from modelswap.engine import SwapEngine
from modelswap.stub_engine import BROKEN_MARKER

MATLAB_NOTE = (
    "MATLAB: open matlab/simulate_swap.m and run it to show the decision boundary "
    "shift and the stream across the swap at t = 50."
)


@dataclass(frozen=True)
class DemoOptions:
    role: str = "classifier_role"
    candidate: str = "models/model_b.pt"
    fast: bool = False
    show_failure: bool = False


@dataclass(frozen=True)
class Step:
    title: str
    command: str
    candidate: str | None = None
    expect_ok: bool = True  # False for steps that should be rejected
    after: str | None = None  # text printed after the step, before the pause


def build_steps(options: DemoOptions) -> list[Step]:
    steps = [
        Step("Initial state: Model A serving", "reset"),
        Step("Show the active model", "status"),
        Step("Compare Model A vs Model B", "compare", candidate=options.candidate),
    ]
    if options.show_failure:
        steps.append(
            Step(
                "Try a broken candidate (the smoke test should reject it)",
                "apply",
                candidate=f"models/model_{BROKEN_MARKER}.pt",
                expect_ok=False,
            )
        )
    steps += [
        Step("Swap: promote Model B", "apply", candidate=options.candidate, after=MATLAB_NOTE),
        Step("Roll back to Model A", "rollback"),
        Step("Confirm the active model", "status"),
    ]
    return steps


def run_demo(
    engine: SwapEngine,
    backend: str,
    options: DemoOptions,
    out: TextIO | None = None,
    pause: Callable[[str], None] | None = None,
) -> int:
    """Run every step. Returns 0 if each step behaved as expected, else 1."""
    out = out or sys.stdout
    pause = pause or _default_pause(options.fast)
    steps = build_steps(options)

    print(f"ML Model Swap demo | role={options.role} | backend={backend}", file=out)
    if backend == "stub":
        print("Backend is the stub: numbers are illustrative until the real engine is wired in.", file=out)

    results: list[tuple[str, bool]] = []
    for number, step in enumerate(steps, 1):
        print(f"\n== Step {number}/{len(steps)}: {step.title} ==", file=out)
        ok = execute(engine, step.command, role=options.role, candidate=step.candidate, out=out)
        passed = ok == step.expect_ok
        results.append((step.title, passed))

        if not passed:
            print(f"[FAIL] step {number} did not behave as expected", file=out)
            break
        if not step.expect_ok:
            print("(expected rejection: the active model did not change)", file=out)
        if step.after:
            print(f"\n{step.after}", file=out)
        if number < len(steps):
            pause("Press Enter for the next step")

    passed_count = sum(1 for _, passed in results if passed)
    print("\nDemo summary:", file=out)
    for title, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {title}", file=out)
    all_passed = passed_count == len(steps)
    print(f"Result: {'PASS' if all_passed else 'FAIL'} ({passed_count}/{len(steps)} steps)", file=out)
    return 0 if all_passed else 1


def _default_pause(fast: bool) -> Callable[[str], None]:
    def pause(message: str) -> None:
        if not fast:
            input(f"\n[{message}]")

    return pause
