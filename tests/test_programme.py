"""Permanent tests for the scenario and the assembled programme. Never deleted.

This is the last place where a modelling mistake can still look like a
result. Two of these tests exist for boundary artefacts that made the
programme infeasible for every schedule - a restriction still in force
after the orders stop, and one that steps rather than ramps - because both
produced a clean "no feasible solution" that says nothing about water.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from faircanal.benchmarks import haughton_filter
from faircanal.closedloop import simulate_closed_loop
from faircanal.control import control_law
from faircanal.delivery import hold_matrix
from faircanal.geometry import uniform_discharge
from faircanal.leximin import LeximinError, solve_leximin, solve_utilitarian, refine
from faircanal.network import corning_cascade
from faircanal.plant import build_plant, horizon_for, response_map
from faircanal.programme import ProgrammeError, assemble, movement_tie_break
from faircanal.scenario import Limits, Scenario, ScenarioError, User, one_user_per_gate

SPEC = haughton_filter(3)
BLOCKS = 6
LEAD = 2


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


@lru_cache(maxsize=2)
def limits(warm: int = 15):
    net = corning_cascade()
    capacity = tuple(
        uniform_discharge(reach.pool, reach.pool.canal_depth_m)
        for reach in net.reaches
    )
    band = tuple(
        (
            -(reach.pool.canal_depth_m - reach.pool.target_level_m),
            reach.pool.canal_depth_m - reach.pool.target_level_m,
        )
        for reach in net.reaches
    )
    return Limits(
        capacity_m3_s=capacity,
        nominal_m3_s=net.steady_discharges,
        level_band_m=band,
        travel_rate_m3_s=tuple(0.25 * value for value in capacity),
        warm_up_steps=warm,
    )


def scenario(fraction: float, **kwargs):
    net = corning_cascade()
    return one_user_per_gate(
        net,
        BLOCKS,
        limits(),
        source_discharge_m3_s=fraction * net.aggregate_demand,
        lead_blocks=LEAD,
        **kwargs,
    )


@lru_cache(maxsize=8)
def programme(fraction: float):
    return assemble(scenario(fraction), mapping())


@lru_cache(maxsize=8)
def solved(fraction: float):
    return solve_leximin(programme(fraction).ratio)


# ---------------------------------------------------------------------------
# The point where nothing is asked for
# ---------------------------------------------------------------------------


def test_ordering_nothing_is_a_perfect_score_at_the_published_point():
    """Everyone drawing their usual amount is everyone fully served.

    The decision variable is a deviation, so the origin is the canal as
    published. If that point did not score one for every user, or violated
    anything, the whole framing would be wrong and every later number would
    be wrong with it.
    """
    model = programme(1.0)
    zero = np.zeros(model.ratio.n_vars)
    assert model.ratios_at(zero) == pytest.approx(np.ones(model.scenario.n_users))
    assert float((model.ratio.a_ub @ zero - model.ratio.b_ub).max()) <= 1e-9


def test_the_ratio_is_the_delivered_volume_over_the_demand():
    """Two routes to the same fraction, one of them not through the matrix."""
    model = programme(0.7)
    rng = np.random.default_rng(5)
    z = np.clip(
        rng.normal(scale=0.2, size=model.ratio.n_vars),
        [bound[0] for bound in model.ratio.bounds],
        [1.0] * model.ratio.n_vars,
    )
    demands = np.array([user.demand_m3 for user in model.scenario.users])
    assert model.ratios_at(z) == pytest.approx(model.volumes_at(z) / demands, rel=1e-10)


def test_the_programme_and_the_loop_agree():
    """The substituted plant is the plant, checked against a real run."""
    model = programme(0.7)
    rng = np.random.default_rng(9)
    z = rng.uniform(-0.1, 0.1, size=model.ratio.n_vars)
    orders = model.orders_at(z)
    per_node = np.zeros((model.size, model.blocks))
    for index, user in enumerate(model.scenario.users):
        per_node[user.node - 1] += orders[index]
    hold = hold_matrix(model.steps, model.blocks, model.scenario.steps_per_block)
    direct = simulate_closed_loop(
        plant().loop,
        np.zeros(model.size),
        hold @ per_node.T,
        law=control_law(
            list(plant().design_models), plant().weights, horizon=model.steps
        ),
    )
    assert model.levels_at(z) == pytest.approx(direct.levels[: model.steps], abs=1e-10)
    assert model.commands_at(z) == pytest.approx(direct.commanded, abs=1e-9)


# ---------------------------------------------------------------------------
# The headline
# ---------------------------------------------------------------------------


def test_leximin_lifts_the_worst_off_and_pays_for_it():
    """What the paper claims, on the published canal with its own controller.

    At seventy per cent of supply the utilitarian answer serves six users
    in full and leaves one at 0.61 - it pays for its total by taking it all
    from one gate. The lexicographic answer gives every user the same 0.89
    instead. The worst-off rises by about half again, the sum falls by
    under seven per cent, and that is the trade the paper is about, in
    numbers rather than in principle.
    """
    model = programme(0.7)
    lex = solved(0.7).ratios
    util = model.ratios_at(solve_utilitarian(model.ratio))

    assert lex.min() > 1.4 * util.min(), (
        f"the lexicographic minimum is {lex.min():.3f} against a utilitarian "
        f"{util.min():.3f}"
    )
    assert lex.sum() < util.sum()
    assert (util.sum() - lex.sum()) / util.sum() < 0.10
    assert lex.min() > 0.8 and util.min() < 0.7
    assert lex.max() - lex.min() < 1e-6, "the fair answer is not equal after all"
    assert util.max() - util.min() > 0.35, "the utilitarian answer is no longer uneven"


def test_scarcity_lowers_the_fraction_everyone_gets():
    """Less water, lower minimum - monotone, and it has to be."""
    fractions = [solved(level).ratios.min() for level in (1.0, 0.9, 0.7, 0.5)]
    assert fractions == sorted(fractions, reverse=True)
    assert fractions[0] == pytest.approx(1.0, abs=1e-6)
    assert fractions[-1] < 0.85


def test_deep_scarcity_has_no_schedule_at_all():
    """Below a certain supply nothing works, and that is the certificate's job.

    The method's answer to an order it cannot meet is not a bad schedule -
    it is a statement that no schedule exists. Here is the supply level at
    which that starts on this canal, so that the certificate has something
    to be built against.
    """
    with pytest.raises(LeximinError):
        solve_leximin(programme(0.3).ratio)


# ---------------------------------------------------------------------------
# Two boundary artefacts, and why they are not findings
# ---------------------------------------------------------------------------


def test_a_restriction_still_in_force_after_the_orders_stop_is_impossible():
    """The run outlasts the blocks, and over that tail nothing can be done.

    Orders exist only for the blocks. The run continues past the last of
    them so the filter's tail lands inside it, and over that tail the
    deviation decays to nothing whatever anybody ordered. A restriction
    still in force there asks for a reduction no schedule can produce.
    """
    model = programme(0.7)
    profile = model.scenario.source_profile()
    last = model.blocks * model.scenario.steps_per_block
    nominal = model.scenario.nominal_draw_m3_s
    assert profile[last] == pytest.approx(nominal)
    assert profile[-1] == pytest.approx(nominal)
    assert min(profile) < nominal

    # Forcing the restriction across the tail makes it infeasible for anyone.
    stubborn = replace(
        model.scenario, source_ramp_steps=0, source_nominal_until=0
    )
    stretched = Scenario(
        name=stubborn.name,
        network=stubborn.network,
        users=stubborn.users,
        blocks=stubborn.blocks,
        limits=stubborn.limits,
        source_discharge_m3_s=stubborn.source_discharge_m3_s,
        source_nominal_until=0,
        source_ramp_steps=0,
    )
    with pytest.raises(LeximinError):
        solve_leximin(assemble(stretched, mapping()).ratio)


def test_a_restriction_that_steps_is_infeasible_and_a_ramped_one_is_not():
    """A notice that takes effect between two samples is not a water problem.

    The filter between an order and the water cannot change the flow in one
    step, so a source profile that jumps is short at exactly the two steps
    where it jumps - by a few per cent, whatever the schedule. Ramping it
    over one order block, the rate the orders themselves change at, removes
    the artefact and changes nothing else.
    """
    stepped = assemble(scenario(0.7, ), mapping())
    stepped = assemble(
        replace(stepped.scenario, source_ramp_steps=0), mapping()
    )
    with pytest.raises(LeximinError):
        solve_leximin(stepped.ratio)
    assert solved(0.7).ratios.min() > 0.5


# ---------------------------------------------------------------------------
# The pieces of the model
# ---------------------------------------------------------------------------


def test_the_cap_stops_over_delivery_from_scoring():
    model = programme(1.0)
    counts = model.row_counts
    assert counts["ratio capped at one"] == model.scenario.n_users
    solution = solved(1.0)
    assert solution.ratios.max() <= model.scenario.ratio_cap + 1e-9


def test_the_announcement_pins_the_blocks_before_it():
    net = corning_cascade()
    late = one_user_per_gate(
        net,
        BLOCKS,
        limits(),
        source_discharge_m3_s=0.7 * net.aggregate_demand,
        lead_blocks=LEAD,
        announced_block=3,
    )
    model = assemble(late, mapping())
    for index in range(late.n_users):
        for block in range(3):
            assert model.ratio.bounds[index * BLOCKS + block] == (0.0, 0.0)
        assert model.ratio.bounds[index * BLOCKS + 3][0] < 0.0


def test_every_constraint_family_is_present_and_counted():
    counts = programme(0.7).row_counts
    for name in (
        "C2 delivered flow non-negative",
        "C3 volume budget",
        "C5 gate flow inside the reach's conveyance",
        "C7 level inside its band",
        "C6 gate travel rate",
        "C8 source availability",
        "ratio capped at one",
    ):
        assert counts[name] > 0, name
    assert sum(counts.values()) == programme(0.7).ratio.a_ub.shape[0]


def test_the_tie_break_picks_the_smoothest_schedule():
    """Section 7.4: same fractions, least gate movement, one answer."""
    model = programme(0.7)
    base = solved(0.7)
    refined = refine(model.ratio, base, movement_tie_break(model))
    assert refined.refined
    assert refined.tie_break_cost is not None
    assert refined.ratios.min() >= base.ratios.min() - base.eps_sat - 1e-9
    movement = lambda z: float(np.abs(np.diff(model.commands_at(z), axis=0)).sum())
    assert movement(refined.z) <= movement(base.z) + 1e-6


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_demand_of_nothing_is_refused():
    with pytest.raises(ScenarioError, match="undefined"):
        User(name="ghost", node=1, nominal_m3_s=1.0, demand_m3=0.0, window=(0, 10))


def test_a_demand_below_the_floor_is_refused():
    net = corning_cascade()
    tiny = User(
        name="tiny", node=1, nominal_m3_s=0.1, demand_m3=0.5, window=(0, 10)
    )
    with pytest.raises(ScenarioError, match="floor"):
        Scenario(
            name="tiny",
            network=net,
            users=(tiny,),
            blocks=BLOCKS,
            limits=limits(),
            source_discharge_m3_s=1.0,
        )


def test_a_zero_weight_is_refused():
    with pytest.raises(ScenarioError, match="weight"):
        User(
            name="w", node=1, nominal_m3_s=1.0, demand_m3=100.0, window=(0, 10), weight=0.0
        )


def test_a_reach_with_no_room_is_refused():
    with pytest.raises(ScenarioError, match="no room"):
        Limits(
            capacity_m3_s=(5.0,),
            nominal_m3_s=(5.0,),
            level_band_m=((-0.2, 0.2),),
            travel_rate_m3_s=(1.0,),
        )


def test_a_band_that_excludes_the_set_point_is_refused():
    with pytest.raises(ScenarioError, match="set-point"):
        Limits(
            capacity_m3_s=(9.0,),
            nominal_m3_s=(5.0,),
            level_band_m=((0.1, 0.2),),
            travel_rate_m3_s=(1.0,),
        )


def test_a_map_and_a_scenario_of_different_shapes_are_refused():
    net = corning_cascade()
    other = one_user_per_gate(
        net, BLOCKS + 1, limits(), source_discharge_m3_s=8.0, lead_blocks=LEAD
    )
    with pytest.raises(ProgrammeError, match="blocks"):
        assemble(other, mapping())


def test_a_warm_up_that_swallows_the_run_is_refused():
    net = corning_cascade()
    swallowed = one_user_per_gate(
        net,
        BLOCKS,
        replace(limits(), warm_up_steps=horizon_for(BLOCKS)),
        source_discharge_m3_s=8.0,
        lead_blocks=LEAD,
    )
    with pytest.raises(ProgrammeError, match="warm-up"):
        assemble(swallowed, mapping())


def test_the_budget_allowance_reaches_the_row_it_is_meant_for():
    """The knob H5 turns, wired to the constraint it is supposed to move.

    The pre-registered question is whether the method's advantage comes
    from its criterion or simply from releasing more water, and the way to
    answer it is to run the same scan with the volume budget loosened. So
    the allowance has to reach C3 and nothing else: the budget's
    right-hand side moves by exactly the demand times the allowance, and
    every other row stays where it was.
    """
    tight = scenario(0.7)
    loose = scenario(0.7, overshoot=0.5)
    first, second = assemble(tight, mapping()), assemble(loose, mapping())
    assert first.row_counts == second.row_counts

    start = 0
    for name, rows in first.row_counts.items():
        stop = start + rows
        moved = second.ratio.b_ub[start:stop] - first.ratio.b_ub[start:stop]
        if name == "C3 volume budget":
            expected = np.array([0.5 * user.demand_m3 for user in tight.users])
            assert np.allclose(moved, expected)
        else:
            assert np.allclose(moved, 0.0), f"{name} moved and should not have"
        start = stop
