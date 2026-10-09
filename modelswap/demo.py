"""Scripted live demo: the README flow, one step at a time.

Each step runs through commands.execute(), the same path the CLI uses. The demo
pauses for Enter between steps unless fast=True (rehearsal).

    python -m modelswap demo                 # live, waits for Enter
    python -m modelswap demo --fast          # no pauses
    python -m modelswap demo --show-failure  # also shows a rejected swap

The CLI runs the demo on a temporary copy of the registry and state, so the
demo never changes models.yaml or swaps/.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

import yaml

from modelswap.commands import execute
from modelswap.engine import SwapEngine
from modelswap.runtime import DEFAULT_REGISTRY
from modelswap.stub_engine import BROKEN_MARKER

# Not a checkpoint: loading it fails, so the smoke test (or the pre-check) rejects it.
CORRUPT_CHECKPOINT = b"this is not a PyTorch checkpoint"


@dataclass(frozen=True)
class DemoOptions:
    role: str = "classifier"
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


def build_steps(options: DemoOptions, corrupt_candidate: str | None = None) -> list[Step]:
    steps = [
        Step("Initial state: Model A serving", "reset"),
        Step("Show the active model", "status"),
        Step("Compare Model A vs Model B", "compare", candidate=options.candidate),
    ]
    if options.show_failure:
        steps.append(
            Step(
                "Try a corrupt candidate (expect rejection: the active model stays)",
                "apply",
                candidate=corrupt_candidate or f"models/model_{BROKEN_MARKER}.pt",
                expect_ok=False,
            )
        )
    steps += [
        Step("Swap: promote Model B", "apply", candidate=options.candidate),
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
    workdir: Path | None = None,
) -> int:
    """Run every step. Returns 0 if each step behaved as expected, else 1."""
    out = out or sys.stdout
    pause = pause or _default_pause(options.fast)

    corrupt_candidate = None
    if options.show_failure:
        corrupt_candidate = str(_write_corrupt_candidate(workdir))
    steps = build_steps(options, corrupt_candidate)

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


def _write_corrupt_candidate(workdir: Path | None) -> Path:
    """Write a corrupt checkpoint file so the rejected-swap step uses a real bad file."""
    directory = workdir or Path(tempfile.mkdtemp(prefix="modelswap-demo-"))
    path = Path(directory) / f"model_{BROKEN_MARKER}.pt"
    path.write_bytes(CORRUPT_CHECKPOINT)
    return path


def _default_pause(fast: bool) -> Callable[[str], None]:
    def pause(message: str) -> None:
        if not fast:
            input(f"\n[{message}]")

    return pause


def find_available_models(models_dir: Path | None = None) -> list[str]:
    """Finds all .pt checkpoint files in models/ and exports/."""
    search_dirs = [Path("models"), Path("exports")]
    if models_dir and Path(models_dir).is_dir():
        search_dirs.insert(0, Path(models_dir))
    found = []
    for d in search_dirs:
        if d.is_dir():
            for p in d.rglob("*.pt"):
                if "broken" not in p.name.lower():
                    rel = str(p).replace("\\", "/")
                    if rel not in found:
                        found.append(rel)
    return sorted(found)


def get_preset_roles(registry_path: Path | None = None) -> list[dict]:
    """Retrieves predefined roles from models.yaml registry."""
    reg = Path(registry_path or DEFAULT_REGISTRY)
    presets = []
    if reg.is_file():
        try:
            with open(reg, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            roles = data.get("roles", {})
            for rname, rcfg in roles.items():
                curr = rcfg.get("current") or rcfg.get("initial") or "models/model_a.pt"
                cand = rcfg.get("candidate") or "models/model_b.pt"
                kind = rcfg.get("kind", rcfg.get("architecture", "ML model"))
                presets.append({
                    "role": rname,
                    "kind": kind,
                    "current": curr,
                    "candidate": cand,
                })
        except Exception:
            pass
    if not presets:
        presets = [
            {"role": "classifier", "kind": "mlp_2layer", "current": "models/model_a.pt", "candidate": "models/model_b.pt"},
            {"role": "sentiment", "kind": "text_classification", "current": "models/text/bert_base_uncased.pt", "candidate": "models/text/roberta_base.pt"},
        ]
    return presets


def run_interactive_menu(
    engine: SwapEngine,
    backend: str,
    out: TextIO | None = None,
    workdir: Path | None = None,
    default_role: str = "classifier",
    default_candidate: str | None = None,
    registry_path: Path | None = None,
) -> int:
    """
    Interactive Demo Console:
    1. Prompts user to select from existing models / roles to swap.
    2. Prompts user with available commands (status, compare, apply, rollback, reset, demo, exit).
    """
    out = out or sys.stdout
    print("==================================================================", file=out)
    print("          ML MODEL SWAP - INTERACTIVE DEMO CONSOLE               ", file=out)
    print("==================================================================", file=out)
    print(f"Backend: {backend} | Engine initialized", file=out)

    presets = get_preset_roles(registry_path)
    models_found = find_available_models()

    while True:
        print("\nAvailable Model Configurations:", file=out)
        for idx, p in enumerate(presets, 1):
            print(f"  [{idx}] {p['role']} ({p['kind']})", file=out)
            print(f"      From (Current)  : {p['current']}", file=out)
            print(f"      To   (Candidate): {p['candidate']}", file=out)

        custom_opt_idx = len(presets) + 1
        print(f"  [{custom_opt_idx}] Custom selection (choose from detected .pt models)", file=out)
        print("  [0] Exit console (or 'q' / 'exit')", file=out)

        try:
            choice = input(f"\nSelect model pair [1-{custom_opt_idx} or 0]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=out)
            return 0

        if choice in ("0", "q", "quit", "exit"):
            print("Exiting ML Model Swap demo.", file=out)
            return 0

        selected_role = default_role
        selected_candidate = default_candidate or "models/model_b.pt"
        selected_current = "models/model_a.pt"

        if choice.isdigit() and 1 <= int(choice) <= len(presets):
            p = presets[int(choice) - 1]
            selected_role = p["role"]
            selected_current = p["current"]
            selected_candidate = p["candidate"]
        elif choice.isdigit() and int(choice) == custom_opt_idx:
            if not models_found:
                print("No .pt models found in directory.", file=out)
                continue
            print("\nAvailable model files:", file=out)
            for m_i, m_path in enumerate(models_found, 1):
                print(f"  [{m_i}] {m_path}", file=out)
            try:
                c_from = input(f"Select Current Model (From) [1-{len(models_found)}]: ").strip()
                c_to = input(f"Select Candidate Model (To) [1-{len(models_found)}]: ").strip()
                if not (c_from.isdigit() and c_to.isdigit()):
                    print("Invalid selection.", file=out)
                    continue
                selected_current = models_found[int(c_from) - 1]
                selected_candidate = models_found[int(c_to) - 1]
                if "text" in selected_current or "text" in selected_candidate:
                    selected_role = "sentiment"
                else:
                    selected_role = "classifier"
            except (EOFError, KeyboardInterrupt):
                return 0
        else:
            print("Invalid selection, please try again.", file=out)
            continue

        # Command Menu for the selected pair
        switch_model = False
        while not switch_model:
            print("\n==================================================================", file=out)
            print(f"Active Role       : {selected_role}", file=out)
            print(f"Current Model (A) : {selected_current}", file=out)
            print(f"Candidate Model(B): {selected_candidate}", file=out)
            print("==================================================================", file=out)
            print("Select an action to execute:", file=out)
            print("  [1] status   - View current active model and rollback pointer", file=out)
            print("  [2] compare  - Evaluate candidate vs active (accuracy, loss, latency)", file=out)
            print("  [3] apply    - Atomically swap active pointer to candidate (with smoke test)", file=out)
            print("  [4] rollback - Instantly revert back to previous model", file=out)
            print("  [5] reset    - Reset active model back to initial state", file=out)
            print("  [6] demo     - Run full automated 7-step walkthrough for this pair", file=out)
            print("  [7] switch   - Switch to another model pair", file=out)
            print("  [0] exit     - Exit console (or 'q' / 'exit')", file=out)
            print("------------------------------------------------------------------", file=out)

            try:
                cmd_choice = input("Enter command [0-7]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.", file=out)
                return 0

            if cmd_choice in ("0", "exit", "quit", "q"):
                print("Exiting ML Model Swap demo.", file=out)
                return 0
            elif cmd_choice in ("1", "status"):
                print("\n--- Running: status ---", file=out)
                execute(engine, "status", role=selected_role, out=out)
            elif cmd_choice in ("2", "compare"):
                print(f"\n--- Running: compare --candidate {selected_candidate} ---", file=out)
                execute(engine, "compare", role=selected_role, candidate=selected_candidate, out=out)
            elif cmd_choice in ("3", "apply"):
                print(f"\n--- Running: apply --candidate {selected_candidate} ---", file=out)
                execute(engine, "apply", role=selected_role, candidate=selected_candidate, out=out)
            elif cmd_choice in ("4", "rollback"):
                print("\n--- Running: rollback ---", file=out)
                execute(engine, "rollback", role=selected_role, out=out)
            elif cmd_choice in ("5", "reset"):
                print("\n--- Running: reset ---", file=out)
                execute(engine, "reset", role=selected_role, out=out)
            elif cmd_choice in ("6", "demo"):
                print("\n--- Running full automated demo flow ---", file=out)
                opts = DemoOptions(role=selected_role, candidate=selected_candidate, fast=False, show_failure=True)
                run_demo(engine, backend, opts, out=out, workdir=workdir)
            elif cmd_choice in ("7", "switch"):
                switch_model = True
            else:
                print(f"Unknown command '{cmd_choice}'. Please select 0-7.", file=out)

            if not switch_model and cmd_choice not in ("0", "exit", "quit", "q"):
                try:
                    input("\n[Press Enter to continue...]")
                except (EOFError, KeyboardInterrupt):
                    return 0
