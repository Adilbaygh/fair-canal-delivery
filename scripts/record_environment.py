#!/usr/bin/env python3
"""Record the runtime environment to results/environment.json.

The article reports the machine, the operating system and the library
versions that produced its numbers. This script is the only place those
values come from, so the article and the archive cannot disagree.

The host name is deliberately not recorded: it identifies a person and adds
nothing a reader needs.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal import __version__  # noqa: E402
from faircanal.config import discretisation_settings, solver_settings  # noqa: E402
from faircanal.provenance import repo_root, write_json  # noqa: E402

TRACKED_PACKAGES = ("numpy", "scipy", "matplotlib", "pytest", "highspy")


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in sorted(TRACKED_PACKAGES):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "git not available"
    commit = out.stdout.strip()
    return commit if commit else "no commit yet"


def build_record() -> dict:
    return {
        "package_version": __version__,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": " ".join(platform.python_build()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "solver": solver_settings(),
        "discretisation": discretisation_settings(),
    }


def main() -> int:
    record = build_record()
    out_path = repo_root() / "results" / "environment.json"
    write_json(out_path, record)
    print("written: results/environment.json")
    print(f"  python  {record['python']['version']}")
    print(f"  system  {record['platform']['system']} {record['platform']['release']}")
    for name, version in record["packages"].items():
        print(f"  {name:<11}{version}")
    print(f"  commit  {record['git_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
