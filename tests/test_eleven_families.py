"""Permanent tests for the constraints the programme gained. Never deleted.

Why this file exists
--------------------
An audit found that the programme being solved was not the programme the
article described. Four of its eleven families were absent, weakened or
written on the wrong signal, and one of them - the storage state - was
defended in a docstring by naming a test that had never been written.
Every test here pins one of those, and each one would fail if the
constraint went away again.

What a test in this file is allowed to assert
---------------------------------------------
Nothing that the code under test also computes. The storage rows are
checked against a direct simulation of the closed loop, not against the
recursion that built them; the containment of the free-gate set is checked
by putting the closed loop's own answer into it; the two gate signals are
required to differ before anything is concluded from the difference. A
test whose expected value came out of the same function is not a test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from faircanal.closedloop import simulate_closed_loop
from faircanal.config import (
    ANNOUNCE_BLOCKS,
    BAND_TOLERANCE_M,
    BLOCKS,
    CONVEYANCE_EFFICIENCY,
    DT_PLANT_S,
    GATE_HEAD_M,
    LEAD_BLOCKS,
    LP_METHOD,
    LP_OPTIONS,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
    TRAVEL_FRACTION,
    WARM_UP_STEPS,
)
from faircanal.control import control_law
from faircanal.delivery import FilterSpec
from faircanal.geometry import (
    CORNING_CHECK_FLOWS,
    corning_gate_limits,
    uniform_discharge,
)
from faircanal.network import corning_cascade
from faircanal.plant import build_plant, horizon_for, response_map
from faircanal.programme import assemble
from faircanal.scenario import Limits, ScenarioError, one_user_per_gate
from faircanal.upperbound import free_gate_programme

FILTER = FilterSpec(order=3, cutoff_rad_per_s=3.0e-3, sample_time_s=DT_PLANT_S)


@pytest.fixture(scope="module")
def canal():
    """The instance of the article, built once for the whole file."""
    network = corning_cascade()
    plant = build_plant(network, FILTER)
    steps = horizon_for(BLOCKS, margin=SETTLE_MARGIN_STEPS)
    law = control_law(list(plant.design_models), plant.weights, horizon=steps)
    mapping = response_map(plant, BLOCKS, margin=SETTLE_MARGIN_STEPS, law=law)
    return network, plant, mapping, law, steps


def _limits(network, warm=WARM_UP_STEPS, gates=True):
    full = [
        uniform_discharge(reach.pool, reach.pool.canal_depth_m)
        for reach in network.reaches
    ]
    return Limits(
        capacity_m3_s=tuple(full),
        nominal_m3_s=network.steady_discharges,
        level_band_m=tuple(
            (
                -(reach.pool.canal_depth_m - reach.pool.target_level_m),
                reach.pool.canal_depth_m - reach.pool.target_level_m,
            )
            for reach in network.reaches
        ),
        travel_rate_m3_s=tuple(TRAVEL_FRACTION * value for value in full),
        gate_capacity_m3_s=corning_gate_limits(GATE_HEAD_M) if gates else None,
        warm_up_steps=warm,
    )


def _programme(canal, fraction, warm=WARM_UP_STEPS, gates=True, announced=None):
    network, _, mapping, _, steps = canal
    scenario = one_user_per_gate(
        network,
        BLOCKS,
        _limits(network, warm=warm, gates=gates),
        source_discharge_m3_s=fraction * network.aggregate_demand,
        window=(LEAD_BLOCKS * STEPS_PER_BLOCK, steps - 1),
        lead_blocks=LEAD_BLOCKS,
        announced_block=ANNOUNCE_BLOCKS if announced is None else announced,
        settle_margin=SETTLE_MARGIN_STEPS,
    )
    return assemble(scenario, mapping)


def _some_feasible_point(programme):
    """Any point of the polytope. Feasibility only, so no objective."""
    answer = linprog(
        np.zeros(programme.ratio.n_vars),
        A_ub=programme.ratio.a_ub,
        b_ub=programme.ratio.b_ub,
        bounds=programme.ratio.bounds,
        method=LP_METHOD,
        options=dict(LP_OPTIONS),
    )
    assert answer.status == 0, f"the polytope is empty: {answer.message}"
    return answer.x


def _drive(programme, z, steps):
    """The offtake table a decision produces, for a direct simulation."""
    size = programme.size
    orders = programme.orders_at(z)
    hold = np.repeat(np.eye(BLOCKS), STEPS_PER_BLOCK, axis=0)
    drive = np.zeros((steps, size))
    for index, user in enumerate(programme.scenario.users):
        drive[: BLOCKS * STEPS_PER_BLOCK, user.node - 1] += hold @ orders[index]
    return drive


# ---------------------------------------------------------------------------
# C10: the substitution really is the loop
# ---------------------------------------------------------------------------


def test_the_substituted_loop_is_the_loop(canal):
    """Every signal the programme bounds is the one the loop produces.

    This is what "C10 is substituted, not relaxed" means, and it is the
    only reason the equality rows can be left out: the levels, the
    commands and the applied flows are not approximations of the closed
    loop's, they are the closed loop's, reached by matrix product instead
    of by simulation. Checked against an actual simulation, which shares
    no code with the response map's assembly.
    """
    _, plant, _, law, steps = canal
    programme = _programme(canal, 0.95)
    z = _some_feasible_point(programme)
    direct = simulate_closed_loop(
        plant.loop, np.zeros(programme.size), _drive(programme, z, steps), law=law
    )
    assert programme.levels_at(z) == pytest.approx(direct.levels, abs=1e-9)
    assert programme.commands_at(z) == pytest.approx(direct.commanded, abs=1e-9)
    assert programme.applied_at(z) == pytest.approx(direct.applied, abs=1e-9)


# ---------------------------------------------------------------------------
# C5 and C6: the water, not the request
# ---------------------------------------------------------------------------


def test_the_two_gate_signals_are_not_the_same_signal(canal):
    """Before concluding anything from the difference, measure it.

    If the filter did nothing, putting C5 on one signal or the other
    would be a distinction without a difference and every test below it
    would be theatre.
    """
    _, _, mapping, _, _ = canal
    rng = np.random.default_rng(0)
    orders = rng.uniform(-0.5, 0.5, (mapping.size, BLOCKS))
    gap = np.abs(mapping.commands_at(orders) - mapping.applied_at(orders)).max()
    assert gap > 1.0, "the filter is doing nothing; the two signals coincide"
    commanded = np.diag(mapping.peak_command_gain())
    applied = np.diag(mapping.peak_applied_gain())
    assert (commanded[:-1] > 3.0 * applied[:-1]).all(), (
        "the command is supposed to swing several times harder than the flow"
    )


def test_c5_bounds_the_applied_flow_and_c5_prime_the_command(canal):
    """Each of the two lands on its own signal, at every step.

    The conveyance and the gate limit the water; the identified range
    limits what the controller may be asked for. Written on one signal
    only, the programme would either let the canal pass more than it can
    or refuse schedules for a reason that is not physical.
    """
    programme = _programme(canal, 0.95)
    z = _some_feasible_point(programme)
    limits = programme.scenario.limits

    applied = programme.applied_at(z)
    for node, (low, high) in enumerate(limits.flow_bounds):
        assert applied[:, node].min() >= low - 1e-7
        assert applied[:, node].max() <= high + 1e-7

    commanded = programme.commands_at(z)
    for node, (low, high) in enumerate(limits.command_bounds):
        assert commanded[:, node].min() >= low - 1e-7
        assert commanded[:, node].max() <= high + 1e-7

    assert programme.row_counts["C5 gate flow inside the reach's conveyance"] == (
        2 * programme.steps * programme.size
    )
    assert programme.row_counts["C5' command inside the linear model's range"] == (
        2 * programme.steps * programme.size
    )


def test_the_gate_limit_binds_before_the_conveyance_on_most_reaches(canal):
    """The orifice half of C5 is not decoration on this canal.

    At the declared head the gates of seven of the eight reaches pass
    less than their reaches convey, so leaving them out - as the code did
    while ``gate_capacity`` sat unused - loosens C5 everywhere but the
    head.
    """
    network, _, _, _, _ = canal
    conveyance = [
        uniform_discharge(reach.pool, reach.pool.canal_depth_m)
        for reach in network.reaches
    ]
    gates = corning_gate_limits(GATE_HEAD_M)
    binding = [gate < capacity for capacity, gate in zip(conveyance, gates)]
    assert sum(binding) == 7
    assert binding[-1] is False, "the head structure has no published gate"


def test_the_gate_limits_are_read_in_node_order(canal):
    """Reversed, because the source numbers its pools the other way.

    The check that catches the reversal is arithmetic rather than
    editorial: a gate has to be able to pass what its own reach already
    carries, and read in the source's order the narrowest gate lands on
    the reach with the largest steady discharge.
    """
    network, _, _, _, _ = canal
    gates = corning_gate_limits(GATE_HEAD_M)
    nominal = network.steady_discharges
    for node, (gate, flow) in enumerate(zip(gates, nominal), start=1):
        assert gate > flow, f"node {node}: a gate of {gate} cannot pass {flow}"
    # The published check flows, read from the downstream end, are the
    # nominal discharges of nodes 1 to 7.
    assert nominal[:7] == pytest.approx(tuple(reversed(CORNING_CHECK_FLOWS[:7])))


# ---------------------------------------------------------------------------
# C11: what the warm-up is allowed to touch
# ---------------------------------------------------------------------------


def test_the_warm_up_touches_only_the_level_band(canal):
    """C7 is exempted for the transient; C5, C5' and C6 never are.

    One shared step index used to free all of them, so the conveyance and
    the gate travel rate were unenforced over the first quarter of an
    hour of every published run - which is exactly the window in which the
    feed-forward makes its largest move.
    """
    warm_free = _programme(canal, 0.95, warm=0).row_counts
    warmed = _programme(canal, 0.95, warm=WARM_UP_STEPS).row_counts
    assert set(warm_free) == set(warmed)
    for name in warm_free:
        if name == "C7 level inside its band":
            assert warmed[name] < warm_free[name]
        else:
            assert warmed[name] == warm_free[name], (
                f"the warm-up changed {name}, and it has no business there"
            )


def test_the_level_band_covers_every_sample_a_run_produces(canal):
    """A run of K steps has K+1 levels, and C7 bounds all of them.

    The last used to be dropped, which left the level at the end of the
    horizon free.
    """
    programme = _programme(canal, 0.95, warm=0)
    assert programme.levels_at(np.zeros(programme.ratio.n_vars)).shape == (
        programme.steps + 1,
        programme.size,
    )
    assert programme.row_counts["C7 level inside its band"] == (
        2 * (programme.steps + 1) * programme.size
    )


# ---------------------------------------------------------------------------
# C9 and C9': the storage state
# ---------------------------------------------------------------------------


def test_no_water_creation(canal):
    """The storage the programme accounts for is the water that moved.

    The model document requires this check by name and it did not exist.
    Its content is that the storage rows are a mass balance and not a
    slack: what a pool holds at the end of a block is what it held
    before, plus what arrived, minus what left through the gate below it
    and through its own offtakes. Checked against a direct simulation of
    the closed loop, so the recursion that built the rows is not used to
    confirm itself.
    """
    _, plant, mapping, law, steps = canal
    programme = _programme(canal, 0.95)
    z = _some_feasible_point(programme)
    run = simulate_closed_loop(
        plant.loop, np.zeros(programme.size), _drive(programme, z, steps), law=law
    )

    size = programme.size
    lags = mapping.transport_lag
    measured = np.zeros((size, BLOCKS + 1))
    for pool in range(size):
        lag = lags[pool]
        arriving = np.zeros(steps)
        if lag < steps:
            arriving[lag:] = run.applied[: steps - lag, pool]
        leaving = run.applied[:, pool - 1] if pool >= 1 else np.zeros(steps)
        net = (
            CONVEYANCE_EFFICIENCY * arriving - leaving - run.offtake_applied[:, pool]
        ) * DT_PLANT_S
        for block in range(BLOCKS):
            first = block * STEPS_PER_BLOCK
            measured[pool, block + 1] = (
                measured[pool, block] + net[first : first + STEPS_PER_BLOCK].sum()
            )

    assert programme.storage_at(z) == pytest.approx(measured, abs=1e-6)
    assert programme.storage_at(z)[:, 0] == pytest.approx(0.0, abs=1e-12), (
        "a run starts from the state it starts from"
    )


def test_the_storage_state_cannot_refill_itself_each_block(canal):
    """A state accumulates; a per-block slack would not.

    If the rows were written block by block instead of as a running
    total, a pool could give up its whole contents in every one of the
    eight blocks and the programme would make water out of nothing. The
    row for block j has to carry every block before it, and the test for
    that is that the rows are nested rather than independent.
    """
    programme = _programme(canal, 0.95)
    rows = programme.storage_rows.reshape(
        programme.size, BLOCKS + 1, programme.ratio.n_vars
    )
    for pool in range(programme.size):
        for block in range(BLOCKS):
            later = rows[pool, block + 1]
            earlier = rows[pool, block]
            assert np.abs(later - earlier).max() > 0.0, (
                "a block that adds nothing is not a state"
            )
        assert np.abs(rows[pool, 0]).max() == 0.0


def test_c9_prime_ties_the_two_accounts_and_is_not_vacuous(canal):
    """The band holds, and it is close enough to hold something.

    Both halves matter. A band the schedules never approach would be
    decoration; a band they cannot meet would make the programme
    infeasible for a reason that is about the model rather than the
    water, which is why its width was measured rather than chosen.
    """
    programme = _programme(canal, 0.95)
    z = _some_feasible_point(programme)
    area = np.array(programme.response.storage_area)
    storage = programme.storage_at(z)

    levels = programme.levels_at(z)
    mean_level = np.zeros((programme.size, BLOCKS + 1))
    for pool in range(programme.size):
        for block in range(BLOCKS + 1):
            first = min(block * STEPS_PER_BLOCK, programme.steps + 1 - STEPS_PER_BLOCK)
            mean_level[pool, block] = levels[
                first : first + STEPS_PER_BLOCK, pool
            ].mean()

    gap = np.abs(storage - area[:, None] * mean_level) / area[:, None]
    assert gap.max() <= BAND_TOLERANCE_M + 1e-9
    assert gap.max() > 0.1 * BAND_TOLERANCE_M, (
        "the two accounts agree far better than the band allows; either the "
        "band is decoration or the measurement behind it has gone stale"
    )


def test_the_band_uses_the_area_the_pool_integrates_at(canal):
    """Backwater, not water surface. They are not the same number.

    The model document wrote the surface area. The pool's integrator gain
    says otherwise, and on this canal the two differ by up to three
    quarters, so the choice is not cosmetic.
    """
    from faircanal.plant import storage_areas

    _, plant, mapping, _, _ = canal
    backwater = np.array(mapping.storage_area)
    surface = np.array(storage_areas(plant, "surface"))
    assert (backwater < surface).all()
    assert (surface / backwater).max() > 1.5


# ---------------------------------------------------------------------------
# C4: notice has to arrive before the shortage does
# ---------------------------------------------------------------------------


def test_the_announcement_must_precede_the_restriction(canal):
    """Refused with a reason, rather than solved into an empty polytope.

    Freezing the orders until the block the restriction takes effect in
    leaves the users no notice at all, and with a filter between an order
    and the water no schedule exists from 95 per cent downwards. That
    infeasibility would read as a finding about the canal.
    """
    network, _, _, _, steps = canal
    with pytest.raises(ScenarioError, match="told about the shortage"):
        one_user_per_gate(
            network,
            BLOCKS,
            _limits(network),
            source_discharge_m3_s=0.95 * network.aggregate_demand,
            window=(LEAD_BLOCKS * STEPS_PER_BLOCK, steps - 1),
            lead_blocks=LEAD_BLOCKS,
            announced_block=LEAD_BLOCKS,
            settle_margin=SETTLE_MARGIN_STEPS,
        )


def test_the_first_blocks_hold_the_announced_order(canal):
    """C4 pins the deviation, so the nominal order stands - not zero.

    Written as ``v = 0`` it would shut the offtake before the
    announcement, which is a different and much stronger constraint than
    the causality it is supposed to encode.
    """
    programme = _programme(canal, 0.95)
    bounds = programme.ratio.bounds
    for index, user in enumerate(programme.scenario.users):
        for block in range(BLOCKS):
            low, high = bounds[index * BLOCKS + block]
            if block < user.announced_block:
                assert (low, high) == (0.0, 0.0)
            else:
                assert low == pytest.approx(-user.nominal_m3_s)
                assert high == pytest.approx(
                    user.max_order_m3_s - user.nominal_m3_s
                )


# ---------------------------------------------------------------------------
# M1: still a bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [1.00, 0.95, 0.50])
def test_the_free_gate_set_contains_the_closed_loop(canal, fraction):
    """Otherwise M1 is not an upper bound on anything.

    The free-gate programme has to accept every schedule the closed loop
    can run, together with the commands, flows and levels the loop
    produces for it. It once did not: C6 there was written on the command
    while the substituted programme bounded the applied flow, and because
    the command swings several times harder the "upper" bound was the
    tighter of the two.
    """
    _, plant, _, _, steps = canal
    programme = _programme(canal, fraction)
    z = _some_feasible_point(programme)
    free = free_gate_programme(programme, plant)

    point = np.concatenate(
        [
            z,
            programme.commands_at(z).reshape(-1),
            programme.applied_at(z).reshape(-1),
            programme.levels_at(z)[:steps].reshape(-1),
        ]
    )
    assert free.ratio.a_eq @ point == pytest.approx(free.ratio.b_eq, abs=1e-8)
    slack = np.asarray(free.ratio.a_ub @ point - free.ratio.b_ub).ravel()
    assert slack.max() <= 1e-7
    for value, (low, high) in zip(point, free.ratio.bounds):
        assert low is None or value >= low - 1e-7
        assert high is None or value <= high + 1e-7


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------


def test_the_certificate_can_relax_every_row_except_the_cap(canal):
    """Nine families, and the one that is deliberately not among them.

    The cap on the fraction is not a limit on the canal: relaxing it
    would not fill an order, only let over-delivery count as if it had.
    Everything else is relaxable, including the two rows whose slacks are
    statements about the model rather than about the water - without them
    a programme blocked by the identified range reports that the canal
    cannot do it, which is true and useless.
    """
    from faircanal.certificate import RELAXABLE, families

    programme = _programme(canal, 0.60)
    assembled = {family.name for family in families(programme)}
    assert assembled == set(RELAXABLE)
    assert len(RELAXABLE) == 9
    assert set(programme.row_counts) - set(RELAXABLE) == {"ratio capped at one"}
    for name in (
        "C5' command inside the linear model's range",
        "C9 pool storage between empty and full",
        "C9' storage agrees with the level",
    ):
        assert name in RELAXABLE


def test_the_digest_separates_runs_that_differ_only_in_the_filter(canal):
    """It used to read the order block count as the filter's order.

    So the sensitivity runs shared a digest with the main scan, and the
    article reported the collision with the wrong cause attached to it.
    """
    from faircanal.certificate import _digest

    network, plant, _, _, steps = canal
    programme = _programme(canal, 0.95)
    other_spec = FilterSpec(
        order=4, cutoff_rad_per_s=3.0e-3, sample_time_s=DT_PLANT_S
    )
    other_plant = build_plant(network, other_spec)
    law = control_law(
        list(other_plant.design_models), other_plant.weights, horizon=steps
    )
    other_map = response_map(
        other_plant, BLOCKS, margin=SETTLE_MARGIN_STEPS, law=law
    )
    other = assemble(programme.scenario, other_map)
    assert _digest(programme) != _digest(other)
