"""Permanent tests for the archive reader. Never deleted.

The desktop window reads every number through :mod:`faircanal.archive`, so
whatever this module gets wrong reaches a reader as a printed figure. Two
things are guarded here above all.

**A verdict is never widened.** Five crashes in this project came from code
that treated "the solver said nothing" as "the solver said no". The
vocabulary is checked here, and so is the rule that follows from it: a
value whose verdict does not license a number is rendered as a dash, never
as a zero and never as a blank.

**A claim about the archive is computed from the archive.** The witness
counts and the bound gaps shown in the window are derived here, so a change
to how they are counted fails a test rather than quietly changing a
sentence on screen.

Nothing here imports a GUI toolkit; these tests run on a machine that has
never installed one.
"""

from __future__ import annotations

import json

import pytest

from faircanal import archive as arc


# ---------------------------------------------------------------------------
# The vocabulary, and the rule that follows from it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recorded, expected",
    [
        ("solved", arc.SOLVED),
        ("infeasible", arc.INFEASIBLE),
        ("undecided", arc.UNDECIDED),
        ("level only", arc.LEVEL_ONLY),
        ("level_only", arc.LEVEL_ONLY),
        ("SOLVED", arc.SOLVED),
        ("", arc.MISSING),
        (None, arc.MISSING),
        ("something nobody writes", arc.MISSING),
    ],
)
def test_every_recorded_status_maps_to_one_of_the_five_words(recorded, expected):
    assert arc.verdict_of(recorded) == expected


def test_no_verdict_is_not_infeasible():
    """The distinction the whole study rests on, asserted directly."""
    assert arc.verdict_of("undecided") != arc.verdict_of("infeasible")
    assert arc.UNDECIDED not in arc.NUMERIC_VERDICTS
    assert arc.INFEASIBLE not in arc.NUMERIC_VERDICTS


@pytest.mark.parametrize("status", [arc.INFEASIBLE, arc.UNDECIDED, arc.MISSING, ""])
def test_a_verdict_without_a_number_renders_a_dash(status):
    """Not a zero, not a blank. A zero is a measurement nobody made."""
    assert arc.shown(0.0, status) == arc.DASH
    assert arc.shown(0.7304, status) == arc.DASH
    assert arc.shown(None, status) == arc.DASH


@pytest.mark.parametrize("status", [arc.SOLVED, arc.LEVEL_ONLY])
def test_a_verdict_with_a_number_renders_it(status):
    assert arc.shown(0.7304335596751115, status) == "0.7304"
    assert arc.shown(1, status) == "1.0000"


def test_a_missing_value_is_a_dash_even_under_a_good_verdict():
    """A solved point with no such field still has no number to print."""
    assert arc.shown(None, arc.SOLVED) == arc.DASH


def test_a_difference_is_rendered_with_its_sign():
    """+0.13 and 0.13 read the same at a glance, and one of them is a level."""
    assert arc.signed(0.131) == "+0.1310"
    assert arc.signed(-0.131) == "-0.1310"
    assert arc.signed(None) == arc.DASH


# ---------------------------------------------------------------------------
# Reading a point
# ---------------------------------------------------------------------------


def _write_point(folder, percent: int, payload: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"Q{percent:03d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _solved(worst: float, spread: float, ratios: dict[str, float]) -> dict:
    return {
        "status": "solved",
        "worst": worst,
        "spread": spread,
        "total": sum(ratios.values()),
        "ratios": dict(ratios),
        "seconds": 1.0,
        "n_programmes": 1,
    }


@pytest.fixture()
def tiny(tmp_path):
    """A two-point archive with one of everything the window has to handle."""
    root = tmp_path
    scan = root / "results" / "scan" / "main"
    _write_point(
        scan,
        95,
        {
            "label": "main",
            "fraction": 0.95,
            "filter_order": 3,
            "cutoff_rad_per_s": 3.0e-3,
            "overshoot": 0.0,
            "users": ["a-1-user", "a-2-user"],
            "variants": {
                "B1": _solved(0.90, 0.05, {"a-1-user": 0.90, "a-2-user": 0.95}),
                "B4": {
                    **_solved(0.97, 0.0, {"a-1-user": 0.97, "a-2-user": 0.97}),
                    "detail": {"accuracy_bound": 2.0e-6},
                },
                "B5": _solved(0.94, 0.0, {"a-1-user": 0.94, "a-2-user": 0.94}),
                "M1": {"status": "level only", "worst": 0.970001},
            },
            "certificate": {
                "fulfilled": False,
                "ratios": {"a-1-user": 0.97, "a-2-user": 0.97},
                "shortfalls_m3": {"a-1-user": 12.0},
                "relaxation": {
                    "binding_families": ["C8 source availability"],
                    "slacks": {"C8 source availability": 1.5, "C3 volume budget": 0.0},
                    "units": {"C8 source availability": "m^3/s"},
                },
                "digest": "abc123",
            },
        },
    )
    _write_point(
        scan,
        50,
        {
            "label": "main",
            "fraction": 0.50,
            "filter_order": 3,
            "users": ["a-1-user", "a-2-user"],
            "variants": {
                "B1": {"status": "infeasible", "worst": None},
                "B4": {"status": "undecided", "why": "the solver returned no verdict"},
                "M1": {"status": "infeasible"},
            },
        },
    )
    return arc.Archive(arc.ProjectPaths(root=root))


def test_an_infeasible_point_hands_back_no_ratios_and_no_worst(tiny):
    point = tiny.point("main", 50)
    assert point is not None
    assert point.status("B1") == arc.INFEASIBLE
    assert point.worst("B1") is None
    assert point.ratios("B1") == {}


def test_an_undecided_point_keeps_what_the_run_said_about_it(tiny):
    point = tiny.point("main", 50)
    assert point is not None
    assert point.status("B4") == arc.UNDECIDED
    assert point.worst("B4") is None
    assert "no verdict" in point.why("B4")


def test_the_bound_carries_a_worst_but_no_vector(tiny):
    """M1 answers the first stage and stops; asking for its shares is an error."""
    point = tiny.point("main", 95)
    assert point is not None
    assert point.status("M1") == arc.LEVEL_ONLY
    assert point.worst("M1") == pytest.approx(0.970001)
    assert point.ratios("M1") == {}


def test_every_number_can_name_the_file_it_came_from(tiny):
    point = tiny.point("main", 95)
    assert point is not None
    assert point.source == "results/scan/main/Q095.json"
    assert point.key_source("variants.B4.worst").endswith("-> variants.B4.worst")


def test_the_solved_levels_and_the_cliff_come_from_the_files(tiny):
    assert tiny.percents("main") == (95, 50)
    assert tiny.solved_percents("main") == (95,)
    assert tiny.cliff("main") == 95
    assert tiny.undecided_points("main") == ((50, "B4"),)


def test_slacks_are_ordered_by_what_is_actually_missing(tiny):
    point = tiny.point("main", 95)
    assert point is not None
    first = point.slacks[0]
    assert first[0] == "C8 source availability"
    assert first[1] == pytest.approx(1.5)
    assert first[2] == "m^3/s"


# ---------------------------------------------------------------------------
# What the window says about the archive
# ---------------------------------------------------------------------------


def test_the_bound_is_attained_against_the_runs_own_accuracy(tiny):
    """Not against zero.

    The staged procedure decides saturation with a tolerance and writes the
    resulting accuracy into its own file. A gap smaller than that is not a
    gap the method claims, and calling it strict would report a distinction
    that does not exist.
    """
    gaps = arc.bound_gaps(tiny, "main")
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.percent == 95
    assert gap.gap == pytest.approx(1.0e-6, abs=1e-9)
    assert gap.tolerance == pytest.approx(2.0e-6)
    assert gap.attained


def test_a_real_gap_is_not_hidden_by_the_tolerance(tmp_path):
    root = tmp_path
    _write_point(
        root / "results" / "scan" / "main",
        50,
        {
            "users": ["a-1-user"],
            "variants": {
                "B4": {
                    "status": "solved",
                    "worst": 0.7304,
                    "spread": 0.0,
                    "ratios": {"a-1-user": 0.7304},
                    "detail": {"accuracy_bound": 8.0e-6},
                },
                "M1": {"status": "level only", "worst": 0.7593},
            },
        },
    )
    gap = arc.bound_gaps(arc.Archive(arc.ProjectPaths(root=root)), "main")[0]
    assert gap.gap == pytest.approx(0.0289, abs=1e-4)
    assert not gap.attained


def test_the_dominance_witness_needs_every_user_to_do_better(tiny):
    """Theorem 1's instance, and it is not counted loosely.

    Both criteria have to sit inside the method's own accuracy of zero
    spread, and the proposed one has to give *every* user strictly more. A
    witness that holds for most users is not the situation the theorem
    describes.
    """
    witnesses = arc.spread_witnesses(tiny, "main")
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.kind == arc.DOMINATED
    assert witness.users_strictly_better == witness.users_total == 2


def test_a_smaller_spread_with_a_worse_worst_off_is_a_different_witness(tmp_path):
    root = tmp_path
    _write_point(
        root / "results" / "scan" / "main",
        50,
        {
            "users": ["a-1-user", "a-2-user"],
            "variants": {
                "B4": {
                    **_solved(0.73, 0.09, {"a-1-user": 0.73, "a-2-user": 0.82}),
                    "detail": {"accuracy_bound": 2.0e-6},
                },
                "B5": _solved(0.72, 0.03, {"a-1-user": 0.72, "a-2-user": 0.78}),
            },
        },
    )
    witnesses = arc.spread_witnesses(arc.Archive(arc.ProjectPaths(root=root)), "main")
    assert [item.kind for item in witnesses] == [arc.DIFFERENT_QUESTION]


def test_the_advantage_range_can_leave_full_supply_out(tiny, tmp_path):
    """At full supply every criterion fills every order.

    Including that point drags the reported range down to zero for a reason
    that has nothing to do with the criterion, and it disagrees with the
    published sensitivity table, which reports the scarce range.
    """
    root = tmp_path / "two"
    scan = root / "results" / "scan" / "main"
    _write_point(scan, 100, {
        "users": ["a-1-user"],
        "variants": {
            "B1": _solved(1.0, 0.0, {"a-1-user": 1.0}),
            "B4": _solved(1.0, 0.0, {"a-1-user": 1.0}),
        },
    })
    _write_point(scan, 95, {
        "users": ["a-1-user"],
        "variants": {
            "B1": _solved(0.90, 0.0, {"a-1-user": 0.90}),
            "B4": _solved(0.97, 0.0, {"a-1-user": 0.97}),
        },
    })
    archive = arc.Archive(arc.ProjectPaths(root=root))
    everything = arc.advantage(archive, "main")
    scarce = arc.advantage(archive, "main", scarce_only=True)
    assert [pct for pct, _ in everything] == [100, 95]
    assert [pct for pct, _ in scarce] == [95]
    assert arc.range_text([value for _, value in scarce], digits=3, sign=True) == "+0.070"


def test_ranges_and_operable_text_read_the_way_the_tables_read():
    assert arc.range_text([]) == arc.DASH
    assert arc.range_text([0.5, 0.5]) == "0.5000"
    assert arc.range_text([0.5, 0.75]) == "0.5000–0.7500"
    assert arc.operable_text([]) == arc.DASH
    assert arc.operable_text([100]) == "100% only"
    assert arc.operable_text([50, 75, 100]) == "50–100%"


def test_the_reader_says_which_expected_files_are_absent(tmp_path):
    """An empty archive is reported as empty, not as an archive of zeroes."""
    archive = arc.Archive(arc.ProjectPaths(root=tmp_path))
    missing = archive.missing()
    assert "results/environment.json" in missing
    assert "DATA/inputs.json" in missing
    assert "results/scan/" in missing
    assert archive.labels() == ()
    assert archive.cliff("main") is None


# ---------------------------------------------------------------------------
# Against the published archive, when there is one
# ---------------------------------------------------------------------------


def _published() -> arc.Archive:
    return arc.Archive(arc.ProjectPaths.discover())


def test_the_published_archive_reads_end_to_end():
    """The one test that touches the real files.

    Skipped rather than failed when the scan has not been run, because a
    fresh clone has no results yet and that is not an error.
    """
    archive = _published()
    if not archive.labels():
        pytest.skip("no scan has been run in this checkout")
    assert arc.MAIN_LABEL in archive.labels()
    for label in archive.labels():
        for point in archive.points(label):
            for code in arc.CODES:
                verdict = point.status(code)
                assert verdict in {
                    arc.SOLVED, arc.INFEASIBLE, arc.UNDECIDED,
                    arc.LEVEL_ONLY, arc.MISSING,
                }
                if verdict not in arc.NUMERIC_VERDICTS:
                    assert point.worst(code) is None, (
                        f"{label} Q{point.percent} {code}: a worst-off share "
                        f"under the verdict {verdict!r}"
                    )


def test_the_published_bound_is_never_below_what_was_achieved():
    """Theorem 2, checked against every point the scan wrote.

    The free-gate programme relaxes the closed loop, so its answer cannot
    be the smaller one. A negative gap would mean the embedding assumption
    has broken - which happened once in this project - and it should stop
    the suite rather than appear on screen.
    """
    archive = _published()
    if not archive.labels():
        pytest.skip("no scan has been run in this checkout")
    for label in archive.labels():
        for gap in arc.bound_gaps(archive, label):
            assert gap.gap >= -gap.tolerance, (
                f"{label} Q{gap.percent}: the bound is below the achieved value "
                f"by {-gap.gap:g}"
            )
