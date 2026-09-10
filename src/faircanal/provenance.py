"""Provenance helpers: repository root, checksums, deterministic writing.

Two rules are enforced here rather than remembered:

* no absolute path that carries a user name ever reaches a published file -
  paths are resolved relative to the repository root, which is found from
  this file's own location or from the FAIRCANAL_ROOT environment variable;
* every text artefact is written with "\n" pinned, keys sorted and a
  trailing newline, so two correct runs produce identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

#: Environment variable that overrides the detected repository root.
ENV_ROOT = "FAIRCANAL_ROOT"

_CHUNK = 1 << 20


def repo_root() -> Path:
    """Return the repository root as an absolute path.

    Detected from this file's location (src/faircanal/provenance.py), or
    taken from FAIRCANAL_ROOT when that variable is set.
    """
    override = os.environ.get(ENV_ROOT)
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def rel_to_root(path: str | os.PathLike[str]) -> str:
    """Return *path* relative to the repository root, in POSIX form."""
    return Path(path).resolve().relative_to(repo_root()).as_posix()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write *text* to *path* with "\n" pinned. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    assert target.is_file(), f"write_text failed to create {target}"
    return target


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Write *payload* as deterministic JSON: sorted keys, indent 2, "\n".

    ``allow_nan`` is off, which is the point of this function existing.
    Python writes NaN and Infinity into JSON happily and reads them back
    happily, so a file carrying them looks fine until somebody opens it
    with a strict reader - jq, jsonlite, most of the JavaScript world - and
    is told the archive is malformed. Refusing here turns that into an
    error at the moment the value is produced, where it can still be
    understood, instead of a year later in somebody else's toolchain.
    """
    try:
        text = json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    except ValueError as error:
        raise ValueError(
            f"refusing to write {path}: {error}. JSON has no NaN and no "
            f"Infinity; a quantity that is not defined is written as null."
        ) from error
    return write_text(path, text)


def write_checksums(
    directory: str | os.PathLike[str],
    out_name: str = "SHA256SUMS.txt",
    skip: Iterable[str] = (),
) -> Path:
    """Write SHA-256 checksums for every file under *directory*.

    Entries are sorted by POSIX-relative path, so the file is byte-identical
    across runs and platforms. The checksum file itself is never hashed.
    """
    base = Path(directory).resolve()
    out_path = base / out_name
    skip_set = {out_name, *skip}

    entries: list[tuple[str, str]] = []
    for item in sorted(base.rglob("*"), key=lambda p: p.as_posix()):
        if not item.is_file():
            continue
        rel = item.relative_to(base).as_posix()
        if rel in skip_set:
            continue
        entries.append((rel, sha256_file(item)))

    entries.sort()
    body = "".join(f"{digest}  {rel}\n" for rel, digest in entries)
    return write_text(out_path, body)
