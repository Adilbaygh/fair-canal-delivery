#!/usr/bin/env python3
"""Entry point for the Fair-Canal-Delivery reproducibility package.

Run with no arguments and the desktop window opens: an appendix to the
article that shows every published result with the file it came from, and
runs the published scripts on request.

Commands
--------
gui     open the window (the default)
env     record the runtime environment to results/environment.json
check   run the permanent test suite
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

#: What to say when the window cannot start because the toolkit is absent.
#: Naming the exact command is the whole point: a reviewer who meets this
#: message should not have to work out what to install.
NO_TOOLKIT = (
    "The desktop window needs PyQt6, which is not installed in this "
    "environment.\n"
    "Install it with:\n"
    "    python -m pip install -r requirements-gui.txt\n"
    "Everything else in this package works without it:\n"
    "    python main.py check      run the permanent test suite\n"
    "    python main.py env        record the runtime environment"
)


def cmd_gui(args: argparse.Namespace) -> int:
    if args.list_pages:
        # Deliberately importable with no GUI toolkit present: the registry
        # is plain data, so this works on a machine that cannot open a window.
        from faircanal.gui import pages  # noqa: PLC0415

        for key in pages.PAGE_KEYS:
            page = pages.page_of(key)
            print(f"{key:<12} {page.section:<10} {page.english}")
        return 0

    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from faircanal.gui.app import launch  # noqa: PLC0415
    except ImportError as error:
        if error.name and error.name.startswith("PyQt6"):
            print(NO_TOOLKIT, file=sys.stderr)
            return 2
        raise

    return launch(
        root=ROOT,
        language=args.language,
        page=args.page,
        screenshot=args.screenshot,
    )


def cmd_env(_args: argparse.Namespace) -> int:
    from scripts.record_environment import main as record_main  # noqa: PLC0415

    return record_main()


def cmd_check(_args: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "pytest"], cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_gui = sub.add_parser("gui", help="open the desktop window (the default)")
    _add_gui_arguments(p_gui)
    p_gui.set_defaults(func=cmd_gui)

    p_env = sub.add_parser("env", help="record the runtime environment")
    p_env.set_defaults(func=cmd_env)

    p_check = sub.add_parser("check", help="run the permanent test suite")
    p_check.set_defaults(func=cmd_check)

    # The same options on the bare command, so "python main.py --language en"
    # works without anybody having to learn that "gui" is the default.
    _add_gui_arguments(parser)
    parser.set_defaults(func=cmd_gui)
    return parser


def _add_gui_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--language",
        choices=("uz", "en"),
        default="uz",
        help="the language the window opens in",
    )
    parser.add_argument(
        "--page",
        default=None,
        help="the page to open (see --list-pages)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="render one page off-screen to this file and exit",
    )
    parser.add_argument(
        "--list-pages",
        action="store_true",
        help="print the page keys and exit; needs no GUI toolkit",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
