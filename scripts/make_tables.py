#!/usr/bin/env python3
"""Build the paper's tables from the result tables, and nothing else.

Same rule as the figures: every number is read from ``results/tables``,
nothing is typed in, and a point with no verdict is printed as an em dash
rather than as a value. Rerun after any scan and the tables follow the
archive.

Three tables, each written twice - LaTeX for the manuscript and Markdown
for drafting and for reading in a terminal:

``table1_scan``
    The scarcity scan: worst-off fraction under each criterion at every
    supply level, with the free-gate bound beside them.
``table2_shares``
    Every user's delivered fraction at the point where the two fairness
    criteria are indistinguishable by spread and yet one dominates the
    other. This is the witness for the Pareto-consistency theorem, so it
    belongs in the paper as numbers and not only as a picture.
``table3_sensitivity``
    The three filter and budget configurations, each reduced to what the
    paper claims about it: where the canal runs out, and whether the
    criterion's advantage survives.

Captions live here, beside the code that fills the table, for the reason
the figure captions do: two of those were wrong within an hour of being
written, and only their being next to the code showed it.

    python scripts/make_tables.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.provenance import repo_root, write_text  # noqa: E402

DASH = "---"           # no verdict, or no schedule: never a number
RULE = "\\RULE"        # a horizontal rule between blocks, not a data row
CODES = ("B1", "B2", "B3", "B4", "B5", "M1")
NAMES = {
    "B1": "B1 unchanged",
    "B2": "B2 one block early",
    "B3": "B3 utilitarian",
    "B4": "B4 lexicographic",
    "B5": "B5 least spread",
    "M1": "M1 bound",
}

CAPTIONS = {
    "table1_scan": (
        "Worst-off delivered fraction $\\min_i r_i$ on the Corning cascade "
        "under each criterion, against source availability as a percentage of "
        "aggregate demand. A dash marks a supply level at which the "
        "substituted programme has no feasible schedule. M1 is the free-gate "
        "upper bound of Theorem 2, computed at its first stage"
    ),
    "table2_shares": (
        "Delivered fraction of every user at 95\\% source availability. Both "
        "B4 and B5 attain a spread of exactly zero, so the spread objective "
        "cannot distinguish them, and yet B4 gives every one of the eight "
        "users strictly more. This is the instance witnessing Theorem 1"
    ),
    "table3_sensitivity": (
        "What each configuration changes and what it does not. The operable "
        "range is the set of supply levels at which a feasible schedule "
        "exists; the last column is the advantage of the lexicographic "
        "criterion over leaving the order unchanged, over that range. The "
        "rows above the rule move the filter or the budget; those below move "
        "an assumption the model rests on, away from its frozen value of "
        "$\\kappa = 1.5$, $\\Delta H$ = 0.10 m or $\\delta$ = 0.15 m. The head "
        "cannot be lowered past 0.076 m, where the narrowest gate can no "
        "longer pass the flow its own reach already carries"
    ),
}


# ---------------------------------------------------------------------------
# Reading the archive
# ---------------------------------------------------------------------------


def _rows(label: str, kind: str) -> list[dict]:
    path = repo_root() / "results" / "tables" / f"{label}_{kind}.csv"
    if not path.exists():
        raise SystemExit(f"no such scan: {path}. Run the scan before the tables.")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worst(label: str) -> dict[str, dict[int, float | None]]:
    out: dict[str, dict[int, float | None]] = defaultdict(dict)
    for row in _rows(label, "summary"):
        value = row["worst"]
        out[row["code"]][int(row["percent"])] = (
            float(value) if value not in ("", None) else None
        )
    return out


def shares(label: str, percent: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _rows(label, "ratios"):
        if int(row["percent"]) == percent:
            out[row["code"]][row["user"]] = float(row["ratio"])
    return out


def spread_of(label: str, percent: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for row in _rows(label, "summary"):
        if int(row["percent"]) == percent:
            value = row["spread"]
            out[row["code"]] = float(value) if value not in ("", None) else None
    return out


def levels(label: str) -> list[int]:
    return sorted({int(row["percent"]) for row in _rows(label, "summary")}, reverse=True)


def edge(table) -> int | None:
    """Lowest supply at which the lexicographic answer exists."""
    have = [p for p, v in table.get("B4", {}).items() if v is not None]
    return min(have) if have else None


def number(value, places: int = 4) -> str:
    return DASH if value is None else f"{value:.{places}f}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(stem: str, header: list[str], body: list[list[str]],
           align: str | None = None) -> list[Path]:
    """One table, written as LaTeX for the journal and Markdown for us."""
    columns = align or ("l" + "r" * (len(header) - 1))
    caption = CAPTIONS[stem]

    tex = [
        "% Generated by scripts/make_tables.py - do not edit by hand.",
        "\\begin{table}[H]",
        f"\\caption{{{caption}.}}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{cell}}}" for cell in header) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        if row and row[0] == RULE:
            tex.append("\\midrule")
            continue
        tex.append(" & ".join(row) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    plain = caption.replace("$\\min_i r_i$", "min r_i").replace("\\%", "%")
    md = [
        f"**{stem}.** {plain}.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    md += [
        "| " + " | ".join(row) + " |"
        for row in body
        if not (row and row[0] == RULE)
    ]
    md += [""]

    out = repo_root() / "results" / "tables"
    written = [
        write_text(out / f"{stem}.tex", "\n".join(tex)),
        write_text(out / f"{stem}.md", "\n".join(md)),
    ]
    return written


# ---------------------------------------------------------------------------
# The three tables
# ---------------------------------------------------------------------------


def table1() -> list[Path]:
    table = worst("main")
    header = ["$Q^{\\mathrm{src}}$ (\\%)"] + [NAMES[c] for c in CODES]
    body, empty = [], []
    for percent in levels("main"):
        row = [number(table[code].get(percent)) for code in CODES]
        if all(cell == DASH for cell in row):
            empty.append(percent)
            continue
        body.append([str(percent)] + row)
    if empty:
        # Four identical rows of dashes say nothing four times. One row
        # naming the levels says the same and leaves the table readable.
        span = ", ".join(str(p) for p in empty)
        body.append([RULE] + [""] * len(CODES))
        body.append([span] + [DASH] * len(CODES))
    return render("table1_scan", header, body)


def table2(percent: int = 95) -> list[Path]:
    rows = shares("main", percent)
    spreads = spread_of("main", percent)
    users = sorted(rows["B4"])
    header = ["User"] + [NAMES[c] for c in ("B1", "B3", "B5", "B4")]
    body = []
    for name in users:
        short = name.replace("corning-", "gate ").replace("-user", "")
        body.append([short] + [f"{rows[c][name]:.4f}" for c in ("B1", "B3", "B5", "B4")])
    body.append([RULE] + [""] * (len(header) - 1))
    body.append(["spread"] + [
        number(spreads.get(c), 4) for c in ("B1", "B3", "B5", "B4")
    ])
    return render("table2_shares", header, body)


def table3() -> list[Path]:
    runs = (
        ("main", "order 3, $\\omega_c$ = 3 mrad/s (pre-registered)"),
        ("cut2e3", "order 3, $\\omega_c$ = 2 mrad/s"),
        ("order4", "order 4, $\\omega_c$ = 3 mrad/s"),
        ("budget", "volume budget +50\\%"),
        (RULE, ""),
        ("kappa125", "outlet headroom $\\kappa = 1.25$"),
        ("kappa200", "outlet headroom $\\kappa = 2.00$"),
        ("head008", "gate rating head $\\Delta H$ = 0.08 m"),
        ("head020", "gate rating head $\\Delta H$ = 0.20 m"),
        ("band010", "storage band $\\delta$ = 0.10 m"),
        ("band025", "storage band $\\delta$ = 0.25 m"),
    )
    header = ["Configuration", "Operable range (\\%)", "Points",
              "$\\min_i r_i$ under B4", "B4 $-$ B1"]
    body = []
    for label, name in runs:
        if label == RULE:
            body.append([RULE] + [""] * (len(header) - 1))
            continue
        table = worst(label)
        low = edge(table)
        got = sorted(p for p, v in table["B4"].items() if v is not None)
        if not got:
            body.append([name, DASH, "0", DASH, DASH])
            continue
        span = f"{low}--100" if low is not None and low < 100 else "100 only"
        values = [table["B4"][p] for p in got]
        gains = [
            table["B4"][p] - table["B1"][p]
            for p in got
            if table["B1"].get(p) is not None and p < 100
        ]
        body.append([
            name,
            span,
            str(len(got)),
            f"{min(values):.4f}--{max(values):.4f}",
            f"+{min(gains):.3f}--+{max(gains):.3f}" if gains else DASH,
        ])
    return render("table3_sensitivity", header, body,
                  align="l" + "r" * (len(header) - 1))


def main() -> int:
    for build in (table1, table2, table3):
        for path in build():
            print(f"  {path.relative_to(repo_root())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
