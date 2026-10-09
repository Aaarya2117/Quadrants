"""modelswap command-line interface.

    python -m modelswap status   [--role R]
    python -m modelswap compare  --candidate PATH [--role R] [--data PATH]
    python -m modelswap apply    --candidate PATH [--role R]
    python -m modelswap rollback [--role R]
    python -m modelswap reset    [--role R]          # restore Model A (rehearsal)
    python -m modelswap demo     [--fast] [--show-failure]

Global options go before the command:
    --backend {auto,stub,real}   default auto (real engine if present, else stub)
    --state-file PATH            stub engine state file

Exit codes: 0 success, 1 operation failed or engine unavailable, 2 usage error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from modelswap.commands import execute
from modelswap.demo import DemoOptions, run_demo
from modelswap.engine import BACKEND_CHOICES, load_engine
from modelswap.results import EngineError
from modelswap.stub_engine import DEFAULT_STATE_FILE

DEFAULT_ROLE = "classifier_role"
DEFAULT_CANDIDATE = "models/model_b.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelswap",
        description="Compare, atomically swap, and roll back ML models with identical architecture.",
    )
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="auto", help="engine backend (default: auto)")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE, help="stub engine state file")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add_role(p: argparse.ArgumentParser) -> None:
        p.add_argument("--role", default=DEFAULT_ROLE, help=f"model role (default: {DEFAULT_ROLE})")

    p = sub.add_parser("status", help="show the active and previous model")
    add_role(p)

    p = sub.add_parser("compare", help="compare the candidate (B) against the active model (A)")
    add_role(p)
    p.add_argument("--candidate", required=True, help="path to candidate model B")
    p.add_argument("--data", default=None, help="validation data path (optional)")

    p = sub.add_parser("apply", help="promote the candidate to active (atomic, smoke-tested)")
    add_role(p)
    p.add_argument("--candidate", required=True, help="path to candidate model B")

    p = sub.add_parser("rollback", help="restore the previous model in one step")
    add_role(p)

    p = sub.add_parser("reset", help="restore the initial registry state (rehearsal)")
    add_role(p)

    p = sub.add_parser("demo", help="run the scripted live demo")
    add_role(p)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE, help=f"candidate model (default: {DEFAULT_CANDIDATE})")
    p.add_argument("--fast", action="store_true", help="no pauses between steps")
    p.add_argument("--show-failure", action="store_true", help="also show a rejected swap")
    return parser


def main(argv: list[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)

    try:
        engine, backend = load_engine(args.backend, args.state_file)
    except EngineError as exc:
        print(f"[FAIL] {exc}", file=out)
        return 1
    if backend == "stub" and args.command != "demo":  # the demo prints its own banner
        print("note: stub backend in use; numbers are illustrative", file=sys.stderr)

    if args.command == "demo":
        options = DemoOptions(
            role=args.role,
            candidate=args.candidate,
            fast=args.fast,
            show_failure=args.show_failure,
        )
        return run_demo(engine, backend, options, out=out)

    ok = execute(
        engine,
        args.command,
        role=args.role,
        candidate=getattr(args, "candidate", None),
        data=getattr(args, "data", None),
        out=out,
    )
    return 0 if ok else 1
