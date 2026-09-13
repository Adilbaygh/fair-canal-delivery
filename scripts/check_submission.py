#!/usr/bin/env python3
"""Check the submission package against the journal's stated requirements.

The checklist in the journal notes is a list of sentences, and a list of
sentences is checked by a person who is tired at midnight on the day of
submission. This checks the machine-checkable half of it and says so out
loud, so that "it meets the requirements" is a command that was run rather
than a claim that was made.

What it checks, and what it cannot
----------------------------------
Everything here is a property of a file on disk: that the figures exist,
are the format the journal takes, carry a resolution at or above the floor
it sets, and have one caption each; that every table was generated and not
hand-edited; that the interactive appendix really is self-contained, since
a supplementary file that silently needs the network is worse than none;
that the archive is valid strict JSON, which it once was not; and that the
environment record names the commit that produced the results.

It cannot check whether the abstract is 200 words until an abstract
exists, and it does not pretend to. Anything that depends on the
manuscript is reported as PENDING rather than passed or failed, and the
exit status ignores pending work: this is a readiness check, not a
gate that lies.

    python scripts/check_submission.py
"""

from __future__ import annotations

import json
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.provenance import repo_root  # noqa: E402

# --- the journal's stated numbers, each with the clause it comes from -----
MIN_DPI = 600                      # "preferably no less than 600 dpi"
FIGURE_FORMATS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MAX_TOTAL_MB = 120                 # "must not exceed 120 MB"
ABSTRACT_WORDS = 200               # "about 200 words maximum"
KEYWORDS_RANGE = (3, 10)           # "three to ten pertinent keywords"

FIGURES = (
    "fig1_scan",
    "fig2_who_goes_short",
    "fig3_filter_decides_feasibility",
    "fig4_budget",
)
TABLES = ("table1_scan", "table2_shares", "table3_sensitivity")
#: Each back-matter statement, with the class command that produces it. The
#: MDPI class takes them as commands, so the printed heading never appears in
#: the source and searching for the heading alone reports them all missing.
BACK_MATTER = (
    ("Author Contributions", "\\authorcontributions"),
    ("Funding", "\\funding"),
    ("Data Availability Statement", "\\dataavailability"),
    ("Conflicts of Interest", "\\conflictsofinterest"),
)

PASS, FAIL, PENDING = "PASS", "FAIL", "PENDING"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, verdict: str, what: str, detail: str = "") -> None:
        self.rows.append((verdict, what, detail))

    def ok(self, condition: bool, what: str, good: str = "", bad: str = "") -> None:
        self.add(PASS if condition else FAIL, what, good if condition else bad)

    def failures(self) -> int:
        return sum(1 for verdict, _, _ in self.rows if verdict == FAIL)

    def print(self) -> None:
        width = max(len(what) for _, what, _ in self.rows) + 2
        for verdict, what, detail in self.rows:
            mark = {PASS: "  ok  ", FAIL: " FAIL ", PENDING: " ....."}[verdict]
            print(f"[{mark}] {what:<{width}} {detail}")


# ---------------------------------------------------------------------------
# Reading a PNG without a dependency
# ---------------------------------------------------------------------------


def png_facts(path: Path) -> dict:
    """Width, height, colour type and dpi, straight from the chunks.

    Written by hand rather than with an imaging library because this is a
    check on the submission package and it should not fail for want of an
    optional install on the machine doing the checking.
    """
    facts: dict = {"dpi": None, "colour": None, "size": None}
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return facts
    offset = 8
    while offset + 8 <= len(raw):
        length, kind = struct.unpack(">I4s", raw[offset:offset + 8])
        body = raw[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            w, h, _depth, colour = struct.unpack(">IIBB", body[:10])
            facts["size"] = (w, h)
            # 2 = truecolour RGB, 6 = RGB with alpha. The journal asks for RGB.
            facts["colour"] = {0: "grey", 2: "RGB", 3: "indexed",
                               4: "grey+a", 6: "RGB+a"}.get(colour, str(colour))
        elif kind == b"pHYs":
            x_per_m, _y, unit = struct.unpack(">IIB", body[:9])
            if unit == 1:                       # metres
                facts["dpi"] = round(x_per_m * 0.0254)
        elif kind == b"IDAT":
            break
        offset += 12 + length
    return facts


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def check_figures(root: Path, report: Report) -> None:
    folder = root / "results" / "figures"
    for stem in FIGURES:
        png = folder / f"{stem}.png"
        if not png.exists():
            report.add(FAIL, f"figure {stem}", "missing - run make_figures.py")
            continue
        facts = png_facts(png)
        dpi, colour, size = facts["dpi"], facts["colour"], facts["size"]
        detail = (
            f"{size[0]}x{size[1]} px, {dpi} dpi, {colour}"
            if size else "unreadable"
        )
        good = (
            png.suffix in FIGURE_FORMATS
            and dpi is not None
            and dpi >= MIN_DPI
            and colour in ("RGB", "RGB+a")
        )
        report.add(PASS if good else FAIL, f"figure {stem}", detail)


def check_captions(root: Path, report: Report) -> None:
    path = root / "results" / "figures" / "captions.md"
    if not path.exists():
        report.add(FAIL, "figure captions", "captions.md missing")
        return
    text = path.read_text(encoding="utf-8")
    found = re.findall(r"\*\*Figure (\d+)\.\*\*", text)
    report.ok(
        len(found) == len(FIGURES),
        "figure captions",
        f"{len(found)} captions for {len(FIGURES)} figures, in the manuscript text",
        f"{len(found)} captions for {len(FIGURES)} figures",
    )


def check_tables(root: Path, report: Report) -> None:
    folder = root / "results" / "tables"
    for stem in TABLES:
        tex, md = folder / f"{stem}.tex", folder / f"{stem}.md"
        if not (tex.exists() and md.exists()):
            report.add(FAIL, f"table {stem}", "missing - run make_tables.py")
            continue
        head = tex.read_text(encoding="utf-8").splitlines()[0]
        generated = "Generated by" in head
        report.ok(
            generated, f"table {stem}",
            "generated from the archive, LaTeX and Markdown",
            "no generator banner - was this edited by hand?",
        )


def check_appendix(root: Path, report: Report) -> None:
    path = root / "results" / "explorer.html"
    if not path.exists():
        report.add(FAIL, "interactive appendix", "missing - run make_explorer.py")
        return
    text = path.read_text(encoding="utf-8")
    # A supplementary file that quietly needs the network is worse than none.
    offenders = []
    if re.search(r"https?://", text):
        offenders.append("absolute URL")
    if re.search(r"<script[^>]+src=", text, re.I):
        offenders.append("external script")
    if re.search(r"<link[^>]+href=", text, re.I):
        offenders.append("external stylesheet")
    if re.search(r"\bfetch\s*\(|XMLHttpRequest", text):
        offenders.append("runtime request")
    size = path.stat().st_size / 1024
    report.ok(
        not offenders, "interactive appendix",
        f"{size:.0f} kB, self-contained, no network",
        "not self-contained: " + ", ".join(offenders),
    )


def check_inputs(root: Path, report: Report) -> None:
    """The other half of reproducibility, and the half that was missing.

    An archive of results whose inputs live only inside Python modules is
    reproducible to a programmer and opaque to everybody else. The reader
    a Data Availability statement is written for should be able to see
    what canal produced these numbers without importing anything.
    """
    folder = root / "DATA"
    csv_path, json_path = folder / "corning_canal.csv", folder / "inputs.json"
    if not (csv_path.exists() and json_path.exists()):
        report.add(FAIL, "model inputs", "DATA/ is empty - run build_data.py")
        return
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    reaches = payload.get("canal", {}).get("reaches", 0)
    rows = len(csv_path.read_text(encoding="utf-8").strip().splitlines()) - 1
    labelled = all(
        payload.get(block, {}).get("provenance")
        for block in ("limits", "filter", "scenario")
    ) and payload.get("canal", {}).get("geometry_source")
    report.ok(
        reaches == rows and rows > 0 and bool(labelled), "model inputs",
        f"{rows} reaches exported, every block cited and labelled",
        "inputs incomplete or a block has no provenance label",
    )


def check_archive(root: Path, report: Report) -> None:
    """Strict JSON, because it once was not and nothing said so."""
    bad = []
    files = sorted((root / "results" / "scan").rglob("*.json"))
    for path in files:
        try:
            json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=lambda name: (_ for _ in ()).throw(
                    ValueError(name)
                ),
            )
        except ValueError as error:
            bad.append(f"{path.name}: {error}")
    report.ok(
        not bad and bool(files), "archive is strict JSON",
        f"{len(files)} result files, all valid",
        "; ".join(bad[:3]) if bad else "no result files found",
    )


def check_environment(root: Path, report: Report) -> None:
    path = root / "results" / "environment.json"
    if not path.exists():
        report.add(FAIL, "environment record", "missing - run record_environment.py")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    commit = payload.get("git_commit", "")
    if not commit:
        report.add(FAIL, "environment record", "records no commit")
        return
    try:
        known = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root, capture_output=True, check=False,
        ).returncode == 0
    except OSError:
        known = False
        report.add(PENDING, "environment commit", "git not available to check")
        return
    report.ok(
        known, "environment commit",
        f"{commit[:12]} is a commit in this repository",
        f"{commit[:12]} is not a commit here - was the record copied?",
    )


def check_size(root: Path, report: Report) -> None:
    total = sum(
        p.stat().st_size
        for p in (root / "results").rglob("*")
        if p.is_file()
    ) / 1024 / 1024
    report.ok(
        total <= MAX_TOTAL_MB, "submission size",
        f"results directory is {total:.1f} MB, under the {MAX_TOTAL_MB} MB cap",
        f"results directory is {total:.1f} MB, over the {MAX_TOTAL_MB} MB cap",
    )


def _abstract_body(text: str) -> "str | None":
    """The abstract, in whichever of the two forms the class uses.

    The MDPI class takes the abstract as an argument, ``\\abstract{...}``,
    not as an environment, so the braces have to be matched rather than
    pattern-matched: the abstract itself contains braces.
    """
    env = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if env:
        return env.group(1)
    start = re.search(r"\\abstract\{", text)
    if not start:
        return None
    depth, index = 1, start.end()
    while index < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        index += 1
    return text[start.end(): index - 1]


def _word_count(body: str) -> int:
    """Words as a reader counts them, not as a regular expression does.

    Commands are dropped rather than counted, and a decimal number is one
    word: splitting on non-word characters turns 0.048 into two, which
    inflates the count of a results-bearing abstract by several words.
    """
    plain = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    for brace in "{}~":
        plain = plain.replace(brace, " ")
    return len([w for w in plain.split() if any(c.isalnum() for c in w)])


def check_manuscript(root: Path, report: Report) -> None:
    """Whatever depends on a manuscript that may not exist yet."""
    candidates = sorted(root.glob("manuscript*.tex"))
    paper = root / "paper"
    if paper.exists():
        candidates += sorted(paper.glob("manuscript*.tex"))
        candidates += sorted(paper.glob("*/manuscript*.tex"))
    if not candidates:
        for what in ("abstract length", "keyword count", "back-matter statements"):
            report.add(PENDING, what, "no manuscript yet")
        return
    text = candidates[0].read_text(encoding="utf-8", errors="replace")

    body = _abstract_body(text)
    if body is not None:
        words = _word_count(body)
        report.ok(
            words <= ABSTRACT_WORDS, "abstract length",
            f"{words} words, at or under {ABSTRACT_WORDS}",
            f"{words} words, over the {ABSTRACT_WORDS}-word limit",
        )
    else:
        report.add(PENDING, "abstract length", "no abstract found")

    keywords = re.search(r"\\keyword[s]?\{(.*?)\}", text, re.S)
    if keywords:
        count = len([k for k in re.split(r"[;,]", keywords.group(1)) if k.strip()])
        low, high = KEYWORDS_RANGE
        report.ok(
            low <= count <= high, "keyword count",
            f"{count} keywords, within {low}-{high}",
            f"{count} keywords, outside {low}-{high}",
        )
    else:
        report.add(PENDING, "keyword count", "no keywords found")

    lowered = text.lower()
    missing = [
        name for name, command in BACK_MATTER
        if name.lower() not in lowered and command not in text
    ]
    unfinished = [
        name for name, command in BACK_MATTER
        if command in text and re.search(
            re.escape(command) + r"\{[^{}]*TODO", text, re.S
        )
    ]
    if missing:
        report.ok(
            False, "back-matter statements",
            "all four present", "missing: " + ", ".join(missing),
        )
    elif unfinished:
        report.add(
            PENDING, "back-matter statements",
            "present but still marked TODO: " + ", ".join(unfinished),
        )
    else:
        report.ok(True, "back-matter statements", "all four present", "")


def main() -> int:
    root = repo_root()
    report = Report()
    check_figures(root, report)
    check_captions(root, report)
    check_tables(root, report)
    check_appendix(root, report)
    check_inputs(root, report)
    check_archive(root, report)
    check_environment(root, report)
    check_size(root, report)
    check_manuscript(root, report)

    print("Submission readiness - Mathematics (MDPI)\n")
    report.print()
    failures = report.failures()
    pending = sum(1 for v, _, _ in report.rows if v == PENDING)
    print()
    if failures:
        print(f"{failures} requirement(s) not met. Fix these before submitting.")
    else:
        print("Every machine-checkable requirement is met.")
    if pending:
        print(f"{pending} check(s) wait on the manuscript and were not counted.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
