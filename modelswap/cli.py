"""modelswap command-line interface.

    python -m modelswap status   [--role R]
    python -m modelswap compare  --candidate PATH [--role R] [--data PATH]
    python -m modelswap apply    --candidate PATH [--role R] [--force]
    python -m modelswap rollback [--role R]
    python -m modelswap reset    [--role R]          # restore Model A (rehearsal)
    python -m modelswap demo     [--fast] [--show-failure]

Global options go before the command:
    --backend {auto,stub,real}   default auto (real engine if present, else stub)
    --registry PATH              models.yaml for the real engine (default: models.yaml)
    --state-file PATH            state file for the stub engine

The demo always runs on a temporary copy of the registry and state, so it does
not change models.yaml or swaps/.

Exit codes: 0 success, 1 operation failed or engine unavailable, 2 usage error.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TextIO

from modelswap.commands import execute
from modelswap.demo import DemoOptions, run_demo, run_interactive_menu
from modelswap.engine import BACKEND_CHOICES, load_engine
from modelswap.results import EngineError
from modelswap.runtime import DEFAULT_REGISTRY
from modelswap.stub_engine import DEFAULT_STATE_FILE

DEFAULT_ROLE = "classifier"
DEFAULT_CANDIDATE = "models/model_b.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelswap",
        description="Compare, atomically swap, and roll back ML models with identical architecture.",
    )
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto", help="engine backend (default: auto)")
    parser.add_argument("--registry", type=Path, default=None,
                        help=f"models.yaml for the real engine (default: {DEFAULT_REGISTRY})")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="state file for the stub engine")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_role(p: argparse.ArgumentParser) -> None:
        p.add_argument("--role", default=DEFAULT_ROLE, help=f"model role (default: {DEFAULT_ROLE})")

    p = sub.add_parser("status", help="show the active and previous model")
    add_role(p)

    p = sub.add_parser("compare", help="compare the candidate (B) against the active model (A)")
    add_role(p)
    p.add_argument("--candidate", required=True, help="path to candidate model B")
    p.add_argument("--data", default=None, help="validation data path (optional)")

    p = sub.add_parser("apply", help="promote the candidate to active (pre-checked, smoke-tested, atomic)")
    add_role(p)
    p.add_argument("--candidate", required=True, help="path to candidate model B")
    p.add_argument("--force", action="store_true",
                   help="apply even if the candidate did not pass the comparison verdict")

    p = sub.add_parser("rollback", help="restore the previous model in one step")
    add_role(p)

    p = sub.add_parser("reset", help="restore the initial registry state (rehearsal)")
    add_role(p)

    p = sub.add_parser("demo", help="run the interactive or scripted demo (on a temporary copy)")
    add_role(p)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE, help=f"candidate model (default: {DEFAULT_CANDIDATE})")
    p.add_argument("--fast", action="store_true", help="no pauses between steps (runs non-interactive)")
    p.add_argument("--show-failure", action="store_true", help="also show a rejected swap in automated flow")
    p.add_argument("-i", "--interactive", action="store_true", help="force interactive model and command selection menu")
    p.add_argument("--auto", action="store_true", help="run non-interactive scripted demo flow instead of menu")
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)

    if args.command == "demo":
        return _run_demo(args, out)

    try:
        engine, backend = load_engine(args.backend, args.state_file, args.registry)
    except EngineError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    if backend == "stub":
        print("note: stub backend in use; numbers are illustrative", file=sys.stderr)

    ok = execute(
        engine,
        args.command,
        role=args.role,
        candidate=getattr(args, "candidate", None),
        data=getattr(args, "data", None),
        out=out,
        force=getattr(args, "force", False),
    )
    return 0 if ok else 1


def _run_demo(args: argparse.Namespace, out: TextIO) -> int:
    """Run the demo on a throwaway copy of the registry and state file."""
    with tempfile.TemporaryDirectory(prefix="modelswap-demo-") as tmp:
        workdir = Path(tmp)
        registry_source = Path(args.registry or DEFAULT_REGISTRY)
        registry_copy = workdir / "models.yaml"
        if registry_source.is_file():
            shutil.copy2(registry_source, registry_copy)
        try:
            engine, backend = load_engine(args.backend, workdir / "state.json", registry_copy, workdir / "swaps")
        except EngineError as exc:
            print(f"[FAIL] {exc}", file=out)
            return 1
        options = DemoOptions(
            role=args.role,
            candidate=args.candidate,
            fast=args.fast,
            show_failure=args.show_failure,
        )

        # Run interactive console by default unless --auto is explicitly passed
        if not getattr(args, "auto", False):
            return run_interactive_menu(
                engine,
                backend,
                out=out,
                workdir=workdir,
                default_role=args.role,
                default_candidate=args.candidate,
                registry_path=registry_copy,
            )
        return run_demo(engine, backend, options, out=out, workdir=workdir)
