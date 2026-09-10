"""Permanent tests for the public/private split. Never deleted.

These guard the three failures that reach readers without any script
noticing: a published file that names a private path (a leak and a dead
pointer at the same time), an absolute local path that identifies the
author's machine, and a credential committed by accident.

The git-history side of the audit needs git and lives in the private audit
tool; what is checkable without git is checked here, on every run.

The private names and the absolute-path markers are assembled from parts at
run time, so this file does not match its own patterns.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Top-level directories that are never published.
PRIVATE_DIRS = (
    "paper",
    "template",
    "references",
    "model",
    "requirements",
    "reviews",
    "review_reports",
    "build",
)

# Public text files are scanned; data and binaries are not.
SCANNED_SUFFIXES = {".py", ".md", ".toml", ".cff", ".txt", ".yml", ".yaml", ".cfg"}

PUBLIC_ROOTS = ("src", "DATA", "results", "tests")
PUBLIC_ROOT_FILES = (
    "README.md",
    "CITATION.cff",
    "pyproject.toml",
    "main.py",
    "requirements.txt",
    "requirements-lock.txt",
)

# Files whose job is to list paths that do not exist yet.
SCAN_EXEMPT = {".gitignore", ".gitattributes"}


def _published_scripts() -> list[Path]:
    """Scripts named in the allowlist that actually exist."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    named = re.findall(r"^!/(scripts/[^\s]+)$", gitignore, flags=re.MULTILINE)
    return [ROOT / name for name in sorted(named) if (ROOT / name).is_file()]


def public_text_files() -> list[Path]:
    files: list[Path] = []
    for name in PUBLIC_ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files.append(path)
    for root_name in PUBLIC_ROOTS:
        root = ROOT / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if not (path.is_file() and path.suffix in SCANNED_SUFFIXES):
                continue
            if any(
                part == "__pycache__" or part.endswith(".egg-info")
                for part in path.parts
            ):
                continue
            files.append(path)
    files.extend(_published_scripts())
    seen: set[Path] = set()
    unique = []
    for path in files:
        if path in seen or path.name in SCAN_EXEMPT:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _private_path_pattern() -> re.Pattern[str]:
    alternatives = "|".join(re.escape(name) for name in PRIVATE_DIRS)
    return re.compile(r"(?<![A-Za-z0-9_.\-/])(" + alternatives + r")/")


def _absolute_path_markers() -> tuple[str, ...]:
    drive = "C:" + chr(92)
    return (
        drive + "Users" + chr(92),
        drive + "Projects" + chr(92),
        "/" + "home" + "/",
        "/" + "Users" + "/",
    )


CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)\b"
    r"\s*[=:]\s*['\"][^'\"]{8,}['\"]"
)


def test_public_files_exist():
    """The bootstrap produced a repository, not an empty directory."""
    files = public_text_files()
    assert files, "no public text files found - was the bootstrap run?"


def test_gitignore_is_an_allowlist():
    """The first non-comment rule must deny everything."""
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    rules = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert rules, ".gitignore has no rules"
    assert rules[0] == "/*", (
        "the first rule must be '/*' - a denylist leaks the first time a "
        f"file is added, found {rules[0]!r}"
    )


@pytest.mark.parametrize("path", public_text_files(), ids=lambda p: p.name)
def test_no_private_path_in_public_file(path: Path):
    """Every path named in published text must resolve to a published file."""
    pattern = _private_path_pattern()
    text = path.read_text(encoding="utf-8", errors="replace")
    offenders = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            offenders.append(f"{path.name}:{lineno}: {match.group(0)}")
    assert not offenders, "published file names a private path:\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", public_text_files(), ids=lambda p: p.name)
def test_no_absolute_local_path(path: Path):
    """No absolute path that carries a user name or a machine layout.

    Matched case-insensitively: pip writes "c:\\projects\\..." in lower case,
    and a case-sensitive check walks straight past it.
    """
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    hits = [m for m in _absolute_path_markers() if m.lower() in text]
    assert not hits, f"{path.name} contains an absolute local path: {hits}"


@pytest.mark.parametrize("path", public_text_files(), ids=lambda p: p.name)
def test_no_credentials(path: Path):
    """No key, token or password literal in a published file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found = CREDENTIAL_PATTERN.findall(text)
    assert not found, f"{path.name} looks like it contains a credential: {found}"


def test_lock_file_pins_versions_only():
    """The lock file must pin versions, not point at somebody's disk.

    "pip freeze" writes the editable install of this package as a line
    naming its absolute source directory. That line carries the author's
    machine layout into a published file and pins nothing, so the lock is
    generated with editable installs excluded.
    """
    lock = ROOT / "requirements-lock.txt"
    if not lock.is_file():
        pytest.skip("requirements-lock.txt has not been generated yet")

    offenders = []
    for lineno, raw in enumerate(lock.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or line.startswith("--editable"):
            offenders.append(f"line {lineno}: {line}")
        elif "==" not in line and not line.startswith("-"):
            offenders.append(f"line {lineno}: not a pinned version: {line}")

    assert not offenders, "requirements-lock.txt is not a clean pin list:\n" + "\n".join(
        offenders
    )
