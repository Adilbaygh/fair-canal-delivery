"""Permanent tests for the provenance helpers. Never deleted.

These are the rules that stop a published artefact from being wrong in a
way nobody notices until it is too late to fix: no absolute path carrying
a user name, identical bytes from two correct runs, and a results file
that a strict reader will actually accept.

The last of those was not a hypothetical. The first scan wrote NaN into
four of its fifteen result files - the points where no schedule exists, so
a structural ceiling is not a quantity that has a value. Python writes NaN
into JSON without complaint and reads it back without complaint, so the
files looked fine. They are not valid JSON, and an archive a strict reader
refuses is not an archive.
"""

from __future__ import annotations

import json
import math

import pytest

from faircanal.provenance import (
    sha256_text,
    write_checksums,
    write_json,
    write_text,
)


def test_a_results_file_that_a_strict_reader_would_refuse_is_refused_first():
    """NaN never reaches a published file, because it cannot be written.

    Not a convention to remember at every call site - an error at the one
    place every artefact goes through. A quantity that is not defined is
    written as null, which JSON has.
    """
    with pytest.raises(ValueError, match="null"):
        write_json("/tmp/faircanal-should-not-exist.json", {"ceiling": math.nan})
    with pytest.raises(ValueError, match="null"):
        write_json("/tmp/faircanal-should-not-exist.json", {"price": math.inf})


def test_null_is_written_and_read_back_as_nothing(tmp_path):
    path = write_json(tmp_path / "record.json", {"ceiling": None, "worst": 0.5})
    text = path.read_text(encoding="utf-8")
    assert "null" in text and "NaN" not in text
    assert json.loads(
        text, parse_constant=lambda name: pytest.fail(f"strict JSON refuses {name}")
    ) == {"ceiling": None, "worst": 0.5}


def test_two_correct_runs_write_the_same_bytes(tmp_path):
    """Determinism, which is what makes a checksum mean anything."""
    payload = {"b": [3, 1, 2], "a": {"z": 1, "y": 2}}
    first = write_json(tmp_path / "one.json", payload).read_bytes()
    second = write_json(tmp_path / "two.json", dict(reversed(payload.items()))).read_bytes()
    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first


def test_the_newline_is_pinned_whatever_the_platform(tmp_path):
    path = write_text(tmp_path / "note.txt", "one\ntwo")
    raw = path.read_bytes()
    assert raw == b"one\ntwo\n"
    assert sha256_text("one\ntwo\n") == sha256_text(raw.decode("utf-8"))


def test_checksums_are_sorted_and_do_not_hash_themselves(tmp_path):
    write_text(tmp_path / "b.txt", "second")
    write_text(tmp_path / "a.txt", "first")
    (tmp_path / "deep").mkdir()
    write_text(tmp_path / "deep" / "c.txt", "third")

    path = write_checksums(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    names = [line.split("  ", 1)[1] for line in lines]
    assert names == sorted(names)
    assert names == ["a.txt", "b.txt", "deep/c.txt"]
    assert "SHA256SUMS.txt" not in names
