#!/usr/bin/env python3
"""Entry point for the Fair-Canal-Delivery reproducibility package.

Subcommands
-----------
env     record the runtime environment to results/environment.json
check   run the permanent test suite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def cmd_env(_args: argparse.Namespace) -> int:
    from scripts.record_environment import main as record_main  # noqa: PLC0415

    return record_main()


def cmd_check(_args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "pytest"], cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env", help="record the runtime environment")
    p_env.set_defaults(func=cmd_env)

    p_check = sub.add_parser("check", help="run the permanent test suite")
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
