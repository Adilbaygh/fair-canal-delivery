"""Permanent tests for the comparisons. Never deleted.

These are the numbers the paper's claim is measured against, so an error
here does not produce a wrong result - it produces a result that looks
right and is not. Most of what follows is therefore not a check that the
code runs but a check that each baseline really is what its name says:
that the utilitarian one really does maximise the total, that the least
spread one really does have the least spread, and that all of them are
solved on the same feasible set as each other.

One test records a fact about the scenario rather than about the code:
bringing a flat order forward in time changes nothing. That is why B2
shifts B1's answer rather than the unchanged order, and the reason is
measured here so it cannot quietly stop being true.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from faircanal.baselines import (
    BaselineError,
    brought_forward,
    l1_nearest,
    leximin,
    min_spread,
    solve_all,
    spread,
    time_shift,
    true_ratios,
    unchanged_order,
    utilitarian,
)
from faircanal.benchmarks import haughton_filter
from faircanal.control import control_law
from faircanal.geometry import uniform_discharge
from faircanal.network import corning_cascade
from faircanal.plant import build_plant, horizon_for, response_map
from faircanal.programme import assemble
from faircanal.scenario import Limits, Scenario, one_user_per_gate

SPEC = haughton_filter(3)
BLOCKS = 6
LEAD = 2
TOL = 1.0e-6


@lru_cache(maxsize=2)
def plant():
    return build_plant(corning_cascade(), SPEC)


@lru_cache(maxsize=2)
def mapping():
    steps = horizon_for(BLOCKS)
    return response_map(
        plant(),
        BLOCKS,
        law=control_law(list(plant().design_models), plant().weights, horizon=steps),
    )


def make_limits() -> Limits:
    net = corning_cascade()
    full = [
        uniform_discharge(reach.pool, reach.pool.canal_depth_m) for reach in net.reaches
    ]
    return Limits(
        capacity_m3_s=tuple(full),
        nominal_m3_s=net.steady_discharges,
        level_band_m=tuple(
            (
                -(reach.pool.canal_depth_m - reach.pool.target_level_m),
                reach.pool.canal_depth_m - reach.pool.target_level_m,
            )
            for reach in net.reaches
        ),
        travel_rate_m3_s=tuple(0.25 * value for value in full),
        warm_up_steps=15,
    )


@lru_cache(maxsize=8)
def model(fraction: float):
    net = corning_cascade()
    scenario = one_user_per_gate(
        net,
        BLOCKS,
        make_limits(),
        source_discharge_m3_s=fraction * net.aggregate_demand,
        lead_blocks=LEAD,
    )
    return assemble(scenario, mapping())


@lru_cache(maxsize=4)
def answers(fraction: float):
    return solve_all(model(fraction))


def violation(programme, z: np.ndarray) -> float:
    """How far outside the polytope a schedule is, in row units."""
    slack = programme.ratio.b_ub - programme.ratio.a_ub @ np.asarray(z)
    return float(max(0.0, -slack.min()))


# ---------------------------------------------------------------------------
# All on one polytope
# ---------------------------------------------------------------------------


def test_every_baseline_lands_inside_the_same_feasible_set():
    """The whole comparison is worthless if they are solved on different sets."""
    programme = model(0.7)
    for code, answer in answers(0.7).items():
        assert violation(programme, answer.z) < TOL, f"{code} is outside the polytope"
        for value, (low, high) in zip(answer.z, programme.ratio.bounds):
            if low is not None:
                assert value >= low - TOL
            if high is not None:
                assert value <= high + TOL


def test_each_baseline_is_best_at_its_own_objective():
    """Each one wins on the thing it optimises, and that is the only check
    of correctness that does not depend on the canal being any particular
    canal: an optimiser that is beaten at its own objective is wrong."""
    found = answers(0.7)
    worst = {code: answer.worst for code, answer in found.items()}
    total = {code: answer.total for code, answer in found.items()}
    norm = {code: float(np.abs(answer.z).sum()) for code, answer in found.items()}
    sigma = {code: answer.spread for code, answer in found.items()}

    assert worst["B4"] >= max(worst.values()) - TOL, worst
    assert total["B3"] >= max(total.values()) - TOL, total
    assert norm["B1"] <= min(norm.values()) + TOL, norm
    assert sigma["B5"] <= min(sigma.values()) + 1.0e-4, sigma


def test_the_lexicographic_answer_cannot_be_beaten_on_the_worst_off():
    """Not a hypothesis - a property of the procedure, kept under guard.

    Whatever the canal or the scarcity, no feasible schedule can lift the
    worst-off fraction above what the first lexicographic stage reaches,
    because that stage maximises exactly that number over exactly this
    set. If this ever fails, the staged procedure is broken, not the
    canal.
    """
    programme = model(0.5)
    best = leximin(programme)
    assert best.detail["stage_levels"][0] == pytest.approx(
        min(best.weighted), abs=best.detail["accuracy_bound"]
    )
    for other in (unchanged_order(programme), utilitarian(programme)):
        assert min(other.weighted) <= min(best.weighted) + best.detail["accuracy_bound"]


# ---------------------------------------------------------------------------
# The time shift, and the reason it shifts what it shifts
# ---------------------------------------------------------------------------


def test_a_flat_order_cannot_be_brought_forward():
    """Why B2 shifts B1's answer and not the unchanged order.

    Every user in this study asks for its usual draw across the whole
    window, so the unchanged order is flat in time, and moving a flat
    thing sideways leaves it where it was. Read literally, B2 would then
    be B1 with an extra solve. The fact is measured here, not assumed,
    so that a later scenario with a shaped order makes this test fail
    loudly rather than silently changing what B2 means.
    """
    programme = model(0.7)
    flat = np.zeros(programme.ratio.n_vars)
    shifted = brought_forward(flat, programme.scenario.n_users, programme.blocks)
    assert np.array_equal(shifted, flat)


def test_bringing_a_schedule_forward_moves_it_and_empties_the_end():
    table = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    moved = brought_forward(table.ravel(), 2, 3, blocks_early=1).reshape(2, 3)
    assert np.array_equal(moved, np.array([[2.0, 3.0, 0.0], [5.0, 6.0, 0.0]]))

    far = brought_forward(table.ravel(), 2, 3, blocks_early=5).reshape(2, 3)
    assert np.array_equal(far, np.zeros((2, 3)))

    with pytest.raises(BaselineError, match="backwards"):
        brought_forward(table.ravel(), 2, 3, blocks_early=-1)


def test_the_time_shift_really_shifts_something():
    """B1's answer is not flat, so B2 is a different schedule from B1."""
    programme = model(0.7)
    first = unchanged_order(programme)
    second = time_shift(programme, first)
    assert second.detail["shift_changed_the_target"], (
        "B1's schedule is flat in time, so B2 is now the same as B1 and the "
        "comparison has lost a column"
    )
    assert np.abs(second.z - first.z).sum() > TOL
    assert second.n_programmes == first.n_programmes + 1


def test_the_shift_is_read_from_the_filter_not_chosen():
    """One block, because the filter's own group delay says so.

    Measured from the filter's coefficients rather than quoted: about
    eleven samples of sixty seconds against a block of fifteen minutes,
    so one block is both the smallest shift an operator working in blocks
    can make and the nearest one to undoing the delay.
    """
    from scipy import signal

    numerator, denominator = SPEC.coefficients()
    _, group = signal.group_delay((numerator, denominator), w=[1e-6, 1e-5])
    delay = float(group[0])
    blocks = delay / model(0.7).scenario.steps_per_block
    assert 0.5 <= blocks <= 1.5, f"{delay} steps is no longer about one block"
    assert round(blocks) == 1


# ---------------------------------------------------------------------------
# The competing criterion
# ---------------------------------------------------------------------------


def test_the_spread_is_the_one_the_competitor_defines():
    """Divided by the number of terms, not by one less - their equation."""
    values = np.array([0.4, 0.8, 1.0])
    assert spread(values) == pytest.approx(float(np.std(values)))
    assert spread(values) != pytest.approx(float(np.std(values, ddof=1)))
    assert spread(np.ones(5)) == pytest.approx(0.0)
    with pytest.raises(BaselineError):
        spread(np.zeros(0))


def test_frank_wolfe_stops_because_it_has_a_bound_not_because_it_ran_out():
    programme = model(0.7)
    answer = min_spread(programme)
    assert answer.detail["converged"], (
        f"the interval was still {answer.detail['certified_width']} wide after "
        f"{answer.detail['iterations']} iterations"
    )
    low, high = answer.detail["sigma_bounds"]
    assert low <= answer.spread <= high + TOL
    assert high - low <= answer.detail["sigma_tol"]
    assert violation(programme, answer.z) < TOL


def test_the_bound_brackets_the_answer_from_both_sides():
    """The interval is an enclosure, not a comfort blanket.

    No feasible schedule anywhere may have a spread below the lower end
    of the reported interval, and the schedule actually returned sits at
    the upper end. Both halves are checked against the other baselines,
    which are feasible schedules and therefore cannot break the bound.
    """
    programme = model(0.5)
    answer = min_spread(programme)
    low, high = answer.detail["sigma_bounds"]
    assert low <= high
    for code, other in answers(0.5).items():
        assert other.spread >= low - TOL, f"{code} beats a bound it cannot beat"
    assert answer.spread <= min(
        other.spread for other in answers(0.5).values()
    ) + answer.detail["sigma_tol"]


def test_a_tighter_start_does_not_change_where_it_ends_up():
    """Convexity, checked rather than recited."""
    programme = model(0.7)
    from_projection = min_spread(programme)
    from_leximin = min_spread(programme, start=leximin(programme, tie_break=False).z)
    assert from_projection.spread == pytest.approx(from_leximin.spread, abs=1.0e-5)


def test_re_mixing_buys_speed_and_not_a_different_answer():
    """Two routes to the same number, and what the slow one costs.

    The correction step is an optimisation of the method, not a change to
    the model, so it must not move the answer. Given the same budget of
    linear programmes the plain method has not finished narrowing its
    interval, so the comparison that means anything is this: the quick
    answer lies inside the interval the plain method certifies, is no
    worse than the number the plain method reached, and got there with a
    far narrower bound and fewer solves.
    """
    programme = model(0.5)
    quick = min_spread(programme)
    plain = min_spread(programme, corrective=False)

    assert quick.detail["converged"]
    assert quick.spread <= plain.spread + TOL
    low, _ = plain.detail["sigma_bounds"]
    assert low - TOL <= quick.spread, "the quick answer beats a bound it cannot beat"

    assert quick.detail["certified_width"] < plain.detail["certified_width"]
    assert quick.n_programmes < plain.n_programmes, (
        f"re-mixing no longer saves anything: {quick.n_programmes} against "
        f"{plain.n_programmes} linear programmes"
    )
    assert quick.detail["corrections"] > 0
    assert plain.detail["corrections"] == 0


def test_a_run_that_cannot_converge_is_refused_rather_than_faked():
    programme = model(0.7)
    with pytest.raises(BaselineError):
        min_spread(programme, max_iterations=0)
    with pytest.raises(BaselineError):
        min_spread(programme, sigma_tol=0.0)


# ---------------------------------------------------------------------------
# When there is enough water
# ---------------------------------------------------------------------------


def test_with_enough_water_every_criterion_agrees():
    """The criteria can only disagree about who goes short."""
    for code, answer in answers(1.0).items():
        assert answer.fulfilled, f"{code} left somebody short at full supply"
        assert answer.spread == pytest.approx(0.0, abs=1.0e-6)


def test_when_there_is_no_schedule_at_all_the_baselines_say_so():
    programme = model(0.3)
    with pytest.raises(BaselineError):
        unchanged_order(programme)


# ---------------------------------------------------------------------------
# Reading a schedule
# ---------------------------------------------------------------------------


def test_fulfilment_is_read_from_the_unweighted_fraction():
    """The same correction the certificate makes, kept here too."""
    base = model(0.7)
    users = tuple(
        replace(user, weight=0.5) if index == 0 else user
        for index, user in enumerate(base.scenario.users)
    )
    scenario = Scenario(
        name="weighted",
        network=base.scenario.network,
        users=users,
        blocks=base.scenario.blocks,
        limits=base.scenario.limits,
        source_discharge_m3_s=base.scenario.source_discharge_m3_s,
        source_nominal_until=base.scenario.source_nominal_until,
    )
    programme = assemble(scenario, mapping())
    answer = leximin(programme)
    assert answer.weighted[0] == pytest.approx(answer.ratios[0] / 0.5, rel=1e-9)
    assert answer.ratios[0] < answer.weighted[0]


def test_the_projection_is_the_nearest_point_and_says_how_near():
    programme = model(0.7)
    z, objective, solved = l1_nearest(programme.ratio)
    assert solved == 1
    assert objective == pytest.approx(float(np.abs(z).sum()), rel=1e-6)
    assert violation(programme, z) < TOL


def test_a_target_of_the_wrong_size_is_refused():
    programme = model(0.7)
    with pytest.raises(BaselineError, match="variables"):
        l1_nearest(programme.ratio, np.zeros(3))


def test_the_ratios_are_what_the_programme_says_they_are():
    programme = model(0.7)
    answer = answers(0.7)["B4"]
    assert np.allclose(answer.ratios, true_ratios(programme.ratio, answer.z))
    assert np.allclose(answer.weighted, programme.ratios_at(answer.z))
    assert answer.total == pytest.approx(float(answer.ratios.sum()))
    assert answer.worst == pytest.approx(float(answer.ratios.min()))


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_the_upper_bound_is_not_solved_on_this_polytope():
    """M1 frees the gate commands, so it is a different programme."""
    with pytest.raises(BaselineError, match="upperbound"):
        solve_all(model(0.7), codes=("B1", "M1"))


def test_an_unknown_code_is_refused():
    with pytest.raises(BaselineError, match="unknown baseline"):
        solve_all(model(0.7), codes=("B1", "B9"))


def test_the_answers_come_back_in_the_reported_order():
    found = solve_all(model(0.7), codes=("B4", "B1"))
    assert list(found) == ["B1", "B4"]
    assert "B1" in found["B1"].line()
