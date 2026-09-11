"""Permanent tests for the scan's bookkeeping. Never deleted.

The scan script has no mathematics in it. What it has is the decision of
which outcome gets written into which field, and that decision has now
been wrong four separate times - each time in the same direction, and each
time discovered only by a run on the author's machine that had already
survived the whole test suite:

* a variant that stopped on a time limit recorded as infeasible;
* a solver status that means "I did not decide" read as "infeasible";
* the free-gate bound given the verdict "infeasible" at a point where it
  was never attempted, on the strength of a different feasible set;
* an exception type that no handler caught, which ended the scan with a
  traceback after two of the four points had already been solved.

None of those is a mistake about the canal. All of them are mistakes about
what a piece of code is entitled to write down, and the reason the suite
missed every one is that the suite tested the solvers and never tested the
bookkeeping. So these tests do not build a canal at all. They replace each
solve with a function that raises exactly what the real one can raise, and
then read the record that comes back.

The rule they pin is one sentence, and it is the rule the four errors
broke: *infeasible* is written only where that solver returned that
verdict on that programme. A solver that stopped, failed, or was never
asked has said nothing, and nothing is what gets recorded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faircanal.baselines import BaselineError
from faircanal.leximin import LeximinError, SolverFailure, SolverLimit, SolverUndecided

ROOT = Path(__file__).resolve().parents[1]


def load():
    """Import the scan script by path; it is a script, not a module."""
    path = ROOT / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_experiments"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scan(monkeypatch):
    """The scan script with the canal taken out from under it.

    Every call that would touch water is replaced. What is left is the
    part under test: the mapping from what a solve did to what the record
    says it did.
    """
    module = load()
    names = ("u1", "u2")
    programme = SimpleNamespace(ratio=SimpleNamespace(names=names), steps=7)

    monkeypatch.setattr(
        module, "one_user_per_gate", lambda *a, **k: SimpleNamespace(
            source_discharge_m3_s=1.0
        )
    )
    monkeypatch.setattr(module, "assemble", lambda *a, **k: programme)

    def answer(worst=0.5):
        return SimpleNamespace(
            z=np.zeros(3),
            ratios=[worst, worst + 0.1],
            objective=worst,
            status=0,
            detail={},
        )

    monkeypatch.setattr(
        module,
        "describe",
        lambda ans, nm, sec: {
            "status": "solved",
            "feasible": True,
            "name": "stub",
            "ratios": {n: float(v) for n, v in zip(nm, ans.ratios)},
            "worst": float(min(ans.ratios)),
            "total": float(sum(ans.ratios)),
            "spread": 0.05,
            "fulfilled": False,
            "n_programmes": 1,
            "seconds": round(sec, 3),
            "detail": {},
        },
    )
    monkeypatch.setattr(module, "report", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "certify",
        lambda *a, **k: SimpleNamespace(fulfilled=False, digest="0" * 16),
    )
    monkeypatch.setattr(
        module,
        "describe_certificate",
        lambda cert: {
            "fulfilled": cert.fulfilled,
            "ratios": {},
            "shortfalls_m3": {},
            "impossible_users": [],
            "relaxation": {"binding_families": []},
            "digest": cert.digest,
        },
    )
    monkeypatch.setattr(module, "solve_leximin", lambda *a, **k: answer())

    module._answer = answer
    return module


def point(module, wanted, record=None, **kw):
    network = SimpleNamespace(aggregate_demand=1.0)
    return module.run_point(
        0.35, network, None, None, None, wanted,
        record if record is not None else {}, 0.0, **kw
    )


def raiser(error):
    def work(*args, **kwargs):
        raise error
    return work


# ---------------------------------------------------------------------------
# The crash itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        SolverLimit("B1 unchanged order: stopped on a limit", ),
        SolverFailure("B1 unchanged order: status 4 (HiGHS Status 0: Not Set)"),
    ],
    ids=["stopped-on-a-limit", "returned-without-deciding"],
)
def test_a_projection_that_does_not_decide_does_not_end_the_scan(scan, error):
    """The bug, in the form it actually arrived in.

    ``unchanged_order`` gained the ability to raise SolverUndecided when
    the status reading was fixed; the handler around it still named only
    BaselineError, so the exception went straight past it and took the
    process with it after two points had already been solved. The scan
    must survive it and say what happened.
    """
    scan.unchanged_order = raiser(error)
    record = point(scan, ("B1", "B2"))
    assert record["variants"]["B1"]["status"] == "undecided"


@pytest.mark.parametrize(
    "error",
    [SolverLimit("stopped"), SolverFailure("did not decide")],
    ids=["limit", "failure"],
)
def test_an_undecided_projection_is_never_written_down_as_infeasible(scan, error):
    """The whole point. No verdict exists, so no verdict is recorded."""
    scan.unchanged_order = raiser(error)
    record = point(scan, ("B1", "B2"))
    for code in ("B1", "B2"):
        assert record["variants"][code]["status"] == "undecided"
        assert record["variants"][code]["feasible"] is None
        assert record["variants"][code]["status"] != "infeasible"


def test_an_undecided_projection_leaves_the_point_to_be_tried_again(scan):
    """Unfinished, not finished-badly.

    A point is re-solved when a variant has no answer, and "undecided" is
    the absence of an answer. If it were recorded as a verdict the point
    would read as done and the missing number would never be filled in -
    which is how a stale wrong verdict survived the fix that corrected it.
    """
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    record = point(scan, ("B1", "B2"))
    scan.unchanged_order = lambda *a, **k: scan._answer(0.4)
    scan.time_shift = lambda *a, **k: scan._answer(0.35)
    again = point(scan, ("B1", "B2"), record=record)
    assert again["variants"]["B1"]["status"] == "solved"
    assert again["variants"]["B2"]["status"] == "solved"


def test_a_solved_point_is_not_solved_twice(scan):
    """The other half of the same rule: an answer is kept."""
    calls = []

    def once(*args, **kwargs):
        calls.append(1)
        return scan._answer(0.4)

    scan.unchanged_order = once
    record = point(scan, ("B1",))
    point(scan, ("B1",), record=record)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# What an undecided projection does and does not take down with it
# ---------------------------------------------------------------------------


def test_only_the_two_variants_that_need_the_projection_lose_their_answer(scan):
    """B2 is B1's answer shifted. The other three are their own programmes.

    Recording five verdicts because one solve failed is the same error as
    recording the bound's verdict because the controller's polytope was
    empty: a claim wider than the evidence. B3, B4 and B5 are solved on
    the same polytope by different programmes, and B5 takes the
    projection only as a starting point, so all three are still asked.
    """
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    scan.utilitarian = lambda *a, **k: scan._answer(0.3)
    scan.leximin = lambda *a, **k: scan._answer(0.6)
    scan.min_spread = lambda *a, **k: scan._answer(0.5)

    record = point(scan, ("B1", "B2", "B3", "B4", "B5"))
    statuses = {code: entry["status"] for code, entry in record["variants"].items()}
    assert statuses == {
        "B1": "undecided",
        "B2": "undecided",
        "B3": "solved",
        "B4": "solved",
        "B5": "solved",
    }


def test_the_least_spread_variant_does_without_a_starting_point(scan):
    """It is handed the projection when there is one, and copes without."""
    seen = {}
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    scan.min_spread = lambda prog, start=None: (
        seen.setdefault("start", start), scan._answer(0.5)
    )[1]
    point(scan, ("B1", "B5"))
    assert seen["start"] is None


def test_an_undecided_projection_writes_no_certificate(scan):
    """A certificate is a claim about the canal, so it needs a verdict.

    The certificate reads the same whether a programme has no schedule or
    the solver merely failed to find out, which is exactly why it must not
    be written in the second case.
    """
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    scan.solve_leximin = raiser(SolverFailure("did not decide either"))
    record = point(scan, ("B1",))
    assert "certificate" not in record


def test_a_certificate_is_still_written_when_the_certificate_solve_decides(scan):
    """One failed solve does not disqualify a different one."""
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    record = point(scan, ("B1",))
    assert record["certificate"]["fulfilled"] is False


# ---------------------------------------------------------------------------
# The empty polytope, which is a different thing and still has to work
# ---------------------------------------------------------------------------


def test_an_empty_polytope_is_recorded_as_infeasible_and_certified(scan):
    """The real verdict is still a verdict, and still gets written."""
    scan.unchanged_order = raiser(BaselineError("no schedule exists"))
    record = point(scan, ("B1", "B2", "B3", "B4", "B5"))
    for code in ("B1", "B2", "B3", "B4", "B5"):
        assert record["variants"][code]["status"] == "infeasible"
        assert record["variants"][code]["feasible"] is False
    assert "certificate" in record


def test_an_empty_polytope_does_not_decide_the_bound(scan):
    """The bound frees the gates, so it is a different feasible set.

    A bound that delivers where the controller cannot is precisely what
    the bound is for, so the one point where that could show up is the
    last place to write down a verdict nobody computed.
    """
    scan.unchanged_order = raiser(BaselineError("no schedule exists"))
    asked = []
    scan.upper_bound_level = lambda *a, **k: (
        asked.append(1), {"status": "level only", "worst": 0.7}
    )[1]
    record = point(scan, ("B1", "M1"), bound_level_only=True)
    assert asked == [1]
    assert record["variants"]["M1"]["status"] != "infeasible"
    assert record["variants"]["B1"]["status"] == "infeasible"


def test_the_half_solved_bound_is_a_status_of_its_own_and_the_tables_know_it(
    scan, tmp_path, monkeypatch
):
    """"level only" is a fourth outcome, not a missing fourth field.

    The first stage of the bound gives a worst-off level and nothing else:
    no per-user fractions, no total, no spread. It is therefore neither
    "solved" (which promises all of those) nor any of the three failures.
    Calling it "solved" to make it look like the rest would put a promise
    in the record that the entry cannot keep - so it says what it is, and
    the one reader that matters is checked here rather than assumed.
    """
    scan.unchanged_order = lambda *a, **k: scan._answer(0.4)
    scan.upper_bound_level = lambda *a, **k: {"status": "level only", "worst": 0.7}
    record = point(scan, ("B1", "M1"), bound_level_only=True)
    entry = record["variants"]["M1"]
    assert entry["status"] == "level only"
    assert entry["feasible"] is True
    assert entry["status"] not in ("infeasible", "undecided")

    # The table builder is the reader that has to cope with the fourth
    # status, so it is run rather than trusted - in a directory of its
    # own, because a test that writes into results/ is a test that
    # publishes made-up numbers.
    monkeypatch.setattr(scan, "repo_root", lambda: tmp_path)
    written = scan.build_tables([record], "unit-test")
    assert all(tmp_path in path.parents for path in written)
    summary = next(path for path in written if "summary" in path.name)
    lines = summary.read_text(encoding="utf-8").splitlines()
    assert any(line.split(",")[1:3] == ["M1", "level only"] for line in lines)


def test_an_undecided_projection_does_not_decide_the_bound_either(scan):
    scan.unchanged_order = raiser(SolverFailure("did not decide"))
    scan.upper_bound_level = lambda *a, **k: {"status": "level only", "worst": 0.7}
    record = point(scan, ("B1", "M1"), bound_level_only=True)
    assert record["variants"]["M1"]["feasible"] is True


# ---------------------------------------------------------------------------
# The three outcomes, kept apart one variant at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        (SolverLimit("stopped on a limit"), "undecided"),
        (SolverFailure("returned without deciding"), "undecided"),
        (BaselineError("no schedule exists"), "infeasible"),
        (LeximinError("no schedule exists"), "infeasible"),
    ],
    ids=["limit", "failure", "baseline-infeasible", "leximin-infeasible"],
)
def test_each_variant_records_what_its_own_solve_actually_said(scan, error, expected):
    scan.unchanged_order = lambda *a, **k: scan._answer(0.4)
    scan.leximin = raiser(error)
    record = point(scan, ("B1", "B4"))
    assert record["variants"]["B4"]["status"] == expected
    assert record["variants"]["B1"]["status"] == "solved"


def test_the_two_undecided_exceptions_are_one_family(scan):
    """A handler that names the family catches both, now and later.

    The crash happened because a handler named a class that did not
    include the exception actually raised. Catching the base class is what
    stops the next one from doing it again.
    """
    assert issubclass(SolverLimit, SolverUndecided)
    assert issubclass(SolverFailure, SolverUndecided)
    assert not issubclass(SolverUndecided, (BaselineError, LeximinError))


# ---------------------------------------------------------------------------
# A run that cannot be performed says so before it spends ten seconds
# ---------------------------------------------------------------------------


def test_a_filter_longer_than_the_horizon_is_refused_not_silently_truncated():
    """The sensitivity run that could not be performed at all.

    Lowering the cut-off to a thousandth of a radian a second lengthens
    the filter's memory to 193 steps against a settling margin of 120, so
    the last block's water would fall off the end of the horizon and the
    volume balance would stop adding up without saying so. The run is
    refused, and the refusal names the number that would make it possible
    rather than leaving the author to find it.
    """
    module = load()
    with pytest.raises(SystemExit) as refused:
        module.build(order=3, cutoff=1.0e-3)
    message = str(refused.value)
    assert "193" in message and "--margin" in message


def test_the_refusal_arrives_before_the_canal_is_built():
    """Fail fast, because the filter alone settles it.

    The check needs the filter and nothing else, so putting it after the
    plant is built spends ten seconds to reach the same refusal. This
    test fails if that ordering is ever reversed.
    """
    module = load()
    built = []
    original = module.build_plant
    module.build_plant = lambda *a, **k: built.append(1) or original(*a, **k)
    try:
        with pytest.raises(SystemExit):
            module.build(order=3, cutoff=1.0e-3)
    finally:
        module.build_plant = original
    assert built == []


def test_a_margin_that_covers_the_filter_is_accepted():
    """And the frozen one covers the pre-registered filter."""
    module = load()
    from faircanal.config import DT_PLANT_S, SETTLE_MARGIN_STEPS
    from faircanal.delivery import FilterSpec, memory_steps

    frozen = FilterSpec(order=3, cutoff_rad_per_s=3.0e-3, sample_time_s=DT_PLANT_S)
    assert memory_steps(frozen) <= SETTLE_MARGIN_STEPS
    # Raising the order lengthens the memory but not past the frozen margin,
    # which is why that sensitivity run needs no flag and the other does.
    fourth = FilterSpec(order=4, cutoff_rad_per_s=3.0e-3, sample_time_s=DT_PLANT_S)
    assert memory_steps(frozen) < memory_steps(fourth) <= SETTLE_MARGIN_STEPS
