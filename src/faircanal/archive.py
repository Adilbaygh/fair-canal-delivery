"""Reading the published archive, with the file every number came from.

The desktop window is an appendix to the article, and an appendix that
computes its own numbers is a second, hidden way of producing results. So
this module is the only way the window reaches a number, and it enforces
two rules that the study is built on.

**A number never travels without its source.** Every accessor returns the
value together with the file and key it was read from, so a card on screen
can print the path underneath it. A caller that has no source has nothing
to show.

**Three verdicts, not two.** A supply level was either solved, or proved
infeasible, or the solver returned no verdict at all. The third is not a
failure to be rounded into the second: it is the honest answer "nobody
computed this", and :func:`shown` refuses to render a number for it. Five
crashes during this study came from code that assumed a verdict somebody
had never computed; the rule is kept here structurally rather than by
memory.

Nothing in this module imports a GUI toolkit, so the archive can be read -
and tested - on a machine that has never installed one.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: The criteria compared at every supply level, in the order they are read.
CODES: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "M1")

#: The proposed criterion, and the two it is measured against most often.
PROPOSED = "B4"
UNCHANGED = "B1"
BOUND = "M1"

#: Solver verdicts. ``LEVEL_ONLY`` is the free-gate bound, which answers the
#: first lexicographic stage and no more, so it has a worst-off fraction and
#: no vector of fractions.
SOLVED = "solved"
INFEASIBLE = "infeasible"
UNDECIDED = "undecided"
LEVEL_ONLY = "level only"
MISSING = "missing"

#: The verdicts that permit a number to be printed.
NUMERIC_VERDICTS = frozenset({SOLVED, LEVEL_ONLY})

#: What a missing value is rendered as. Never a zero, never a blank cell.
DASH = "—"

#: Saturation tolerance of the staged procedure, from the frozen settings.
#: Two worst-off fractions closer than this are not distinguishable by the
#: procedure that produced them, and the window says so rather than
#: reporting a difference the method never claimed.
EPS_SAT = 1.0e-6

#: The scan whose configuration was pre-registered. Other labels are
#: sensitivity runs and each says in its own files what was changed.
MAIN_LABEL = "main"


def verdict_of(status: str | None) -> str:
    """Map a recorded status onto the vocabulary above."""
    if not status:
        return MISSING
    text = str(status).strip().lower()
    if text == SOLVED:
        return SOLVED
    if text == INFEASIBLE:
        return INFEASIBLE
    if text in (LEVEL_ONLY, "level_only"):
        return LEVEL_ONLY
    if text == UNDECIDED:
        return UNDECIDED
    return MISSING


def shown(value: Any, status: str | None, digits: int = 4) -> str:
    """Render a number, or a dash when the verdict does not permit one.

    This is the rule the study is built on, in one function: a supply level
    that was proved infeasible has no worst-off fraction, and one where the
    solver returned no verdict has no worst-off fraction *that anybody
    computed*. Neither is a zero and neither is a blank.
    """
    if verdict_of(status) not in NUMERIC_VERDICTS:
        return DASH
    if value is None:
        return DASH
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return DASH


def fraction(value: Any, digits: int = 4) -> str:
    """Render a fraction that is already known to exist."""
    if value is None:
        return DASH
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return DASH


def signed(value: Any, digits: int = 4) -> str:
    """Render a difference with its sign, so a gain cannot read as a level."""
    if value is None:
        return DASH
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return DASH


def percent_text(value: Any, digits: int = 2) -> str:
    if value is None:
        return DASH
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return DASH


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------

#: Overridden by the same environment variable the library already uses, so
#: the window and the scripts agree on which repository they are looking at.
ENV_ROOT = "FAIRCANAL_ROOT"


@dataclass(frozen=True)
class ProjectPaths:
    """The published directories, found once and passed around."""

    root: Path

    @classmethod
    def discover(cls, start: Path | str | None = None) -> "ProjectPaths":
        override = os.environ.get(ENV_ROOT)
        if start is not None:
            base = Path(start).resolve()
        elif override:
            base = Path(override).resolve()
        else:
            base = Path(__file__).resolve().parents[2]
        return cls(root=base)

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def scan(self) -> Path:
        return self.results / "scan"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def data(self) -> Path:
        return self.root / "DATA"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    def relative(self, path: Path | str) -> str:
        """A path as the reader should see it: relative, POSIX, no machine."""
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return Path(path).name


# ---------------------------------------------------------------------------
# One supply level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One scarcity point: everything the scan wrote about one supply level."""

    label: str
    percent: int
    payload: dict
    path: Path
    root: Path

    # -- provenance ---------------------------------------------------------

    @property
    def source(self) -> str:
        try:
            return self.path.resolve().relative_to(self.root).as_posix()
        except (ValueError, OSError):
            return self.path.name

    def key_source(self, key: str) -> str:
        return f"{self.source} -> {key}"

    # -- the run's own settings --------------------------------------------

    @property
    def filter_order(self) -> int | None:
        return self.payload.get("filter_order")

    @property
    def cutoff_rad_per_s(self) -> float | None:
        return self.payload.get("cutoff_rad_per_s")

    @property
    def overshoot(self) -> float | None:
        return self.payload.get("overshoot")

    @property
    def source_m3_s(self) -> float | None:
        return self.payload.get("source_m3_s")

    @property
    def horizon_steps(self) -> int | None:
        return self.payload.get("horizon_steps")

    @property
    def blocks(self) -> int | None:
        return self.payload.get("blocks")

    @property
    def users(self) -> tuple[str, ...]:
        listed = self.payload.get("users") or []
        return tuple(sorted(str(name) for name in listed))

    # -- the criteria -------------------------------------------------------

    def variant(self, code: str) -> dict:
        return dict(self.payload.get("variants", {}).get(code) or {})

    def status(self, code: str) -> str:
        return verdict_of(self.variant(code).get("status"))

    def name(self, code: str) -> str:
        return str(self.variant(code).get("name") or "")

    def worst(self, code: str) -> float | None:
        block = self.variant(code)
        if verdict_of(block.get("status")) not in NUMERIC_VERDICTS:
            return None
        value = block.get("worst")
        return None if value is None else float(value)

    def spread(self, code: str) -> float | None:
        block = self.variant(code)
        if verdict_of(block.get("status")) != SOLVED:
            return None
        value = block.get("spread")
        return None if value is None else float(value)

    def total(self, code: str) -> float | None:
        block = self.variant(code)
        value = block.get("total")
        return None if value is None else float(value)

    def seconds(self, code: str) -> float | None:
        value = self.variant(code).get("seconds")
        return None if value is None else float(value)

    def programmes(self, code: str) -> int | None:
        value = self.variant(code).get("n_programmes")
        return None if value is None else int(value)

    def ratios(self, code: str) -> dict[str, float]:
        """Delivered fraction per user, or an empty mapping if there is none.

        Empty is the honest answer for an infeasible point and for the
        free-gate bound, which answers the first stage and stops.
        """
        block = self.variant(code)
        if verdict_of(block.get("status")) != SOLVED:
            return {}
        found = block.get("ratios") or {}
        return {str(user): float(value) for user, value in found.items()}

    def why(self, code: str) -> str:
        """What the run said about a point it could not answer."""
        return str(self.variant(code).get("why") or "")

    def accuracy_bound(self, code: str = PROPOSED) -> float | None:
        """How close to the lexicographic optimum the run itself claims to be.

        The staged procedure decides saturation with a tolerance, so its
        answer can sit below the optimum by at most the number of users
        times that tolerance. The run writes the resulting bound into its
        own file, and it is read from there rather than recomputed: a
        difference smaller than this is not a difference the method claims.
        """
        value = (self.variant(code).get("detail") or {}).get("accuracy_bound")
        if value is not None:
            return float(value)
        users = len(self.users)
        return EPS_SAT * users if users else None

    # -- the certificate ----------------------------------------------------

    @property
    def certificate(self) -> dict:
        return dict(self.payload.get("certificate") or {})

    @property
    def binding_families(self) -> tuple[str, ...]:
        relaxation = self.certificate.get("relaxation") or {}
        listed = relaxation.get("binding_families") or []
        return tuple(str(name) for name in listed)

    @property
    def slacks(self) -> list[tuple[str, float, str]]:
        """Elastic relaxation: what each family would have to give, in its unit."""
        relaxation = self.certificate.get("relaxation") or {}
        values = relaxation.get("slacks") or {}
        units = relaxation.get("units") or {}
        rows = [
            (str(name), float(amount), str(units.get(name, "")))
            for name, amount in values.items()
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        return rows

    @property
    def digest(self) -> str:
        return str(self.certificate.get("digest") or "")

    @property
    def determinism(self) -> dict:
        return dict(self.payload.get("determinism") or {})


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[list[str]]
    path: Path
    source: str


@dataclass(frozen=True)
class Figure:
    number: int
    name: str
    path: Path
    caption: str
    exists: bool


#: The figures the article reports, in its own order. The caption is read
#: from the file the drawing script writes beside them, so a caption cannot
#: drift from the picture it belongs to.
FIGURE_NAMES: tuple[str, ...] = (
    "fig1_scan",
    "fig2_who_goes_short",
    "fig3_filter_decides_feasibility",
    "fig4_budget",
)


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------


@dataclass
class Archive:
    """Everything the published run wrote, read on demand and cached."""

    paths: ProjectPaths
    _points: dict[tuple[str, int], Point | None] = field(default_factory=dict)
    _tables: dict[str, Table | None] = field(default_factory=dict)

    # -- labels and points --------------------------------------------------

    def labels(self) -> tuple[str, ...]:
        """Scan labels present, with the pre-registered one first."""
        if not self.paths.scan.is_dir():
            return ()
        found = sorted(
            item.name for item in self.paths.scan.iterdir() if item.is_dir()
        )
        if MAIN_LABEL in found:
            found.remove(MAIN_LABEL)
            found.insert(0, MAIN_LABEL)
        return tuple(found)

    def percents(self, label: str) -> tuple[int, ...]:
        """Supply levels present for a label, richest first."""
        folder = self.paths.scan / label
        if not folder.is_dir():
            return ()
        values: list[int] = []
        for item in folder.glob("Q*.json"):
            digits = item.stem[1:]
            if digits.isdigit():
                values.append(int(digits))
        return tuple(sorted(values, reverse=True))

    def point(self, label: str, percent: int) -> Point | None:
        key = (label, percent)
        if key in self._points:
            return self._points[key]
        path = self.paths.scan / label / f"Q{percent:03d}.json"
        found: Point | None = None
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            if payload:
                found = Point(
                    label=label,
                    percent=percent,
                    payload=payload,
                    path=path,
                    root=self.paths.root,
                )
        self._points[key] = found
        return found

    def points(self, label: str) -> tuple[Point, ...]:
        found = [self.point(label, pct) for pct in self.percents(label)]
        return tuple(item for item in found if item is not None)

    def settings(self, label: str) -> dict:
        """The configuration a label was run with, taken from its own files."""
        for point in self.points(label):
            return {
                "filter_order": point.filter_order,
                "cutoff_rad_per_s": point.cutoff_rad_per_s,
                "overshoot": point.overshoot,
                "blocks": point.blocks,
                "horizon_steps": point.horizon_steps,
            }
        return {}

    # -- what the scan found ------------------------------------------------

    def solved_percents(self, label: str, code: str = PROPOSED) -> tuple[int, ...]:
        return tuple(
            point.percent
            for point in self.points(label)
            if point.status(code) in NUMERIC_VERDICTS
        )

    def cliff(self, label: str, code: str = PROPOSED) -> int | None:
        """The lowest supply level at which a schedule still exists."""
        solved = self.solved_percents(label, code)
        return min(solved) if solved else None

    def undecided_points(self, label: str) -> tuple[tuple[int, str], ...]:
        """Every place the solver returned no verdict, with what it said."""
        found: list[tuple[int, str]] = []
        for point in self.points(label):
            for code in CODES:
                if point.status(code) == UNDECIDED:
                    found.append((point.percent, code))
        return tuple(found)

    # -- tables -------------------------------------------------------------

    def table(self, name: str, limit: int = 400) -> Table | None:
        if name in self._tables:
            return self._tables[name]
        path = self.paths.tables / name
        found: Table | None = None
        if path.is_file():
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.reader(handle))
            except OSError:
                rows = []
            if rows:
                found = Table(
                    header=rows[0],
                    rows=[row for row in rows[1 : limit + 1]],
                    path=path,
                    source=self.paths.relative(path),
                )
        self._tables[name] = found
        return found

    def table_names(self) -> tuple[str, ...]:
        if not self.paths.tables.is_dir():
            return ()
        return tuple(sorted(item.name for item in self.paths.tables.glob("*.csv")))

    # -- figures ------------------------------------------------------------

    def captions(self) -> dict[str, str]:
        """Caption text keyed by figure number, read from the drawn file."""
        path = self.paths.figures / "captions.md"
        if not path.is_file():
            return {}
        found: dict[str, str] = {}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return {}
        for block in text.split("\n\n"):
            stripped = block.strip()
            if not stripped.startswith("**Figure "):
                continue
            head, _, rest = stripped.partition("**")
            head, _, rest = rest.partition("**")
            number = head.replace("Figure", "").strip().rstrip(".")
            found[number] = " ".join(rest.split())
        return found

    def figures(self) -> tuple[Figure, ...]:
        captions = self.captions()
        found: list[Figure] = []
        for index, name in enumerate(FIGURE_NAMES, start=1):
            path = self.paths.figures / f"{name}.png"
            found.append(
                Figure(
                    number=index,
                    name=name,
                    path=path,
                    caption=captions.get(str(index), ""),
                    exists=path.is_file(),
                )
            )
        return tuple(found)

    # -- inputs and environment ---------------------------------------------

    def inputs(self) -> dict:
        path = self.paths.data / "inputs.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def canal_rows(self) -> list[dict[str, str]]:
        path = self.paths.data / "corning_canal.csv"
        if not path.is_file():
            return []
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError:
            return []

    def environment(self) -> dict:
        path = self.paths.results / "environment.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # -- completeness -------------------------------------------------------

    def missing(self) -> list[str]:
        """Published files the window expects and did not find."""
        wanted: list[Path] = [
            self.paths.results / "environment.json",
            self.paths.data / "inputs.json",
            self.paths.data / "corning_canal.csv",
            self.paths.figures / "captions.md",
        ]
        for name in FIGURE_NAMES:
            wanted.append(self.paths.figures / f"{name}.png")
        for label in (MAIN_LABEL,):
            for suffix in ("summary", "ratios", "certificates"):
                wanted.append(self.paths.tables / f"{label}_{suffix}.csv")
        absent = [self.paths.relative(path) for path in wanted if not path.is_file()]
        if not self.labels():
            absent.append("results/scan/")
        return sorted(absent)


# ---------------------------------------------------------------------------
# What the archive says, as questions the article asks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gap:
    """One supply level's distance from the free-gate bound of Theorem 2."""

    percent: int
    bound: float
    achieved: float
    gap: float
    tolerance: float
    attained: bool


def bound_gaps(archive: Archive, label: str = MAIN_LABEL) -> list[Gap]:
    """Theorem 2, point by point: how far the answer is below its own bound.

    ``attained`` is decided against the accuracy the run states for itself,
    not against zero. The staged procedure declares saturation with a
    tolerance, so it never claimed to land closer than that; calling a gap
    of one tolerance "strict" would report a difference the method does not
    have. The tolerance is read from the run's own file.
    """
    rows: list[Gap] = []
    for point in archive.points(label):
        bound = point.worst(BOUND)
        achieved = point.worst(PROPOSED)
        if bound is None or achieved is None:
            continue
        difference = bound - achieved
        tolerance = point.accuracy_bound() or EPS_SAT
        rows.append(
            Gap(
                percent=point.percent,
                bound=bound,
                achieved=achieved,
                gap=difference,
                tolerance=tolerance,
                attained=difference <= tolerance,
            )
        )
    rows.sort(key=lambda row: -row.percent)
    return rows


@dataclass(frozen=True)
class Witness:
    """One supply level at which spread and leximin disagree."""

    percent: int
    kind: str
    spread_proposed: float
    spread_rival: float
    worst_proposed: float
    worst_rival: float
    users_strictly_better: int
    users_total: int


#: Two ways the spread objective and the lexicographic one come apart, and
#: they are different claims, so they are counted separately.
DOMINATED = "dominated"        # equal spread, yet every user does better
DIFFERENT_QUESTION = "different-question"  # smaller spread, worse worst-off


def spread_witnesses(
    archive: Archive,
    label: str = MAIN_LABEL,
    rival: str = "B5",
) -> list[Witness]:
    """Where the least-spread criterion and the lexicographic one disagree.

    Theorem 1 says a translation-invariant dispersion measure cannot tell a
    point from a strictly better one directly above it. That is a statement
    about the measure, proved without any data. What the archive can add is
    an instance: a supply level where both criteria reach a spread that the
    method cannot distinguish from zero, and one of them still gives every
    user strictly more.

    "Cannot distinguish from zero" is decided against the accuracy the run
    states for itself, not against a threshold chosen here and not against
    the four decimals a table happens to print.
    """
    found: list[Witness] = []
    for point in archive.points(label):
        ours = point.ratios(PROPOSED)
        theirs = point.ratios(rival)
        if not ours or not theirs:
            continue
        shared = sorted(set(ours) & set(theirs))
        if not shared:
            continue
        tolerance = point.accuracy_bound() or EPS_SAT
        spread_ours = point.spread(PROPOSED)
        spread_theirs = point.spread(rival)
        worst_ours = point.worst(PROPOSED)
        worst_theirs = point.worst(rival)
        if None in (spread_ours, spread_theirs, worst_ours, worst_theirs):
            continue
        better = sum(1 for user in shared if ours[user] > theirs[user] + tolerance)
        worse = sum(1 for user in shared if ours[user] < theirs[user] - tolerance)

        if (
            spread_ours <= tolerance
            and spread_theirs <= tolerance
            and better == len(shared)
            and worse == 0
        ):
            kind = DOMINATED
        elif spread_theirs < spread_ours - tolerance and worst_ours > worst_theirs + tolerance:
            kind = DIFFERENT_QUESTION
        else:
            continue

        found.append(
            Witness(
                percent=point.percent,
                kind=kind,
                spread_proposed=spread_ours,
                spread_rival=spread_theirs,
                worst_proposed=worst_ours,
                worst_rival=worst_theirs,
                users_strictly_better=better,
                users_total=len(shared),
            )
        )
    found.sort(key=lambda row: -row.percent)
    return found


#: Full supply. Both criteria fill every order there, so the difference
#: between them is zero by construction and says nothing about either.
FULL_SUPPLY = 100


def advantage(
    archive: Archive,
    label: str = MAIN_LABEL,
    over: str = UNCHANGED,
    scarce_only: bool = False,
) -> list[tuple[int, float]]:
    """What the lexicographic criterion gains the worst-off user, point by point.

    ``scarce_only`` drops full supply. At full supply every criterion fills
    every order, so the gain is zero for a reason that has nothing to do
    with the criterion; including it in a reported range drags the low end
    to zero and makes the comparison look weaker than it is. The published
    sensitivity table reports the scarce range, and so does this.
    """
    rows: list[tuple[int, float]] = []
    for point in archive.points(label):
        if scarce_only and point.percent >= FULL_SUPPLY:
            continue
        ours = point.worst(PROPOSED)
        theirs = point.worst(over)
        if ours is None or theirs is None:
            continue
        rows.append((point.percent, ours - theirs))
    rows.sort(key=lambda row: -row[0])
    return rows


def range_text(values: Sequence[float], digits: int = 4, sign: bool = False) -> str:
    """A closed range, or a single value when the two ends coincide."""
    if not values:
        return DASH
    low, high = min(values), max(values)
    render = signed if sign else fraction
    if abs(high - low) < 5 * 10 ** (-digits - 1):
        return render(low, digits)
    return f"{render(low, digits)}–{render(high, digits)}"


def operable_text(percents: Iterable[int]) -> str:
    """The supply levels a configuration can be operated over, as a range."""
    values = sorted(percents)
    if not values:
        return DASH
    if len(values) == 1:
        return f"{values[0]}% only"
    return f"{values[0]}–{values[-1]}%"
