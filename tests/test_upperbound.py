"""Permanent tests for the upper bound. Never deleted.

M1 is the only thing in this package that is allowed to be better than the
proposed method, and the whole point of it is the comparison, so it has to
be the *same canal*. That is what most of these tests check: not that the
free-gate programme solves, but that its plant is the plant, its filter is
the filter, and the schedule the proposed method produces is feasible
inside it. If that containment ever fails, M1 stops being a bound and
becomes a different experiment with a misleading name.

One test exists for a mistake that would not have announced itself. The
substituted programme carries the level at the *start* of each step; the
recursion naturally produces the level at the *end*. The two differ by one
step of level change - about a centimetre on this canal - so a bound
applied to the wrong one would still have looked entirely reasonable.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from faircanal.baselines import leximin, upper_bound
from faircanal.benchmarks import haughton_filter
from faircanal.control import control_law
from faircanal.delivery import apply_filter
from faircanal.geometry import uniform_discharge
from faircanal.network import corning_cascade
from faircanal.plant import build_plant, horizon_for, response_map
from faircanal.pool import simulate_level
from faircanal.programme import assemble
from faircanal.scenario import Limits, one_user_per_gate
from faircanal.upperbound import UpperBoundError, free_gate_programme

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


@lru_cache(maxsize=4)
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
def free(fraction: float):
    return free_gate_programme(model(fraction), plant())


def offtakes(programme, z: np.ndarray) -> np.ndarray:
    """What the users draw at each gate, step by step, from an order."""
    size, blocks = programme.response.size, programme.blocks
    per_gate = np.asarray(programme.spread @ z).reshape(size, blocks)
    return programme.response.delivery @ per_gate.T


def run_plant(programme, commands: np.ndarray, z: np.ndarray):
    """Drive the plant by hand: filter the commands, then run the pools.

    Deliberately written from :mod:`faircanal.pool` and
    :mod:`faircanal.delivery` rather than from the free-gate matrices, so
    that agreement between the two means something.
    """
    net = programme.scenario.network
    steps, size = programme.steps, programme.response.size
    applied = np.column_stack(
        [apply_filter(SPEC, commands[:, index]) for index in range(size)]
    )
    drawn = offtakes(programme, z)
    levels = np.zeros((steps, size))
    for reach, pool in zip(net.reaches, plant().plant_models):
        index = reach.node - 1
        children = net.children(reach.node)
        outflow = (
            np.sum([applied[:, child - 1] for child in children], axis=0)
            if children
            else np.zeros(steps)
        )
        levels[:, index] = simulate_level(
            pool, applied[:, index], outflow, drawn[:, index]
        )[:steps]
    return applied, levels


def vector(programme, z: np.ndarray, commands: np.ndarray) -> np.ndarray:
    applied, levels = run_plant(programme, commands, z)
    return np.concatenate([z, commands.reshape(-1), applied.reshape(-1), levels.reshape(-1)])


def residuals(bound, x: np.ndarray):
    equality = np.abs(bound.ratio.a_eq @ x - bound.ratio.b_eq).max()
    inequality = float((bound.ratio.a_ub @ x - bound.ratio.b_ub).max())
    low = np.array([-np.inf if b[0] is None else b[0] for b in bound.ratio.bounds])
    high = np.array([np.inf if b[1] is None else b[1] for b in bound.ratio.bounds])
    return equality, inequality, float(max((low - x).max(), (x - high).max()))


# ---------------------------------------------------------------------------
# It has to be the same canal
# ---------------------------------------------------------------------------


def test_the_free_gate_plant_reproduces_the_closed_loop():
    """The two ways of writing the same plant, held against each other.

    The substituted programme folds the loop into rows; this one writes it
    as recursions. Driving the recursions with the controller's own
    commands has to give the controller's own levels, to machine
    precision. Anything looser would mean the bound is being computed on a
    slightly different canal, and a slightly different canal is enough to
    make a bound not a bound.
    """
    programme = model(0.7)
    z = leximin(programme, tie_break=False).z
    commands = programme.commands_at(z)
    _, levels = run_plant(programme, commands, z)

    # The programme's level index k is the level at the *start* of step k,
    # and it now carries one sample more than the run has steps, so the
    # comparison takes the first ``steps`` of them.
    assert np.abs(levels - programme.levels_at(z)[: programme.steps]).max() < 1.0e-12


def test_the_level_is_the_one_at_the_start_of_the_step():
    """The off-by-one that would not have announced itself.

    Reading the level at the end of the step instead of the start is a
    real difference on this canal - about a centimetre - but not a loud
    one: every number would still have looked plausible and the band would
    quietly have been applied to the wrong quantity. This measures the
    size of that difference, so the test above is known to discriminate
    between the two readings rather than passing for free.
    """
    programme = model(0.7)
    z = leximin(programme, tie_break=False).z
    net = programme.scenario.network
    applied, _ = run_plant(programme, programme.commands_at(z), z)
    drawn = offtakes(programme, z)

    shifted = np.zeros((programme.steps, programme.response.size))
    for reach, pool in zip(net.reaches, plant().plant_models):
        index = reach.node - 1
        children = net.children(reach.node)
        outflow = (
            np.sum([applied[:, child - 1] for child in children], axis=0)
            if children
            else np.zeros(programme.steps)
        )
        shifted[:, index] = simulate_level(
            pool, applied[:, index], outflow, drawn[:, index]
        )[1:]

    gap = np.abs(shifted - programme.levels_at(z)[: programme.steps]).max()
    assert gap > 1.0e-4, "the two readings of the level no longer differ"
    assert gap < 0.1, f"a step of level change of {gap} m is not a step of level change"


def test_the_proposed_schedule_is_feasible_here():
    """Containment, which is the whole reason M1 may be called a bound.

    Take the schedule the proposed method produces, plug in the
    controller's own commands as the free variable, and the result must
    satisfy every constraint of the free-gate programme. Then the
    free-gate optimum cannot be below the proposed method's, whatever the
    canal or the scarcity.
    """
    for fraction in (1.0, 0.7):
        programme, bound = model(fraction), free(fraction)
        z = leximin(programme, tie_break=False).z
        x = vector(programme, z, programme.commands_at(z))

        equality, inequality, outside = residuals(bound, x)
        assert equality < 1.0e-9, f"Q={fraction}: the plant does not hold, {equality}"
        assert inequality < TOL, f"Q={fraction}: a row is violated by {inequality}"
        assert outside < TOL, f"Q={fraction}: a bound is violated by {outside}"
        assert np.allclose(bound.ratio.ratios_at(x), programme.ratios_at(z))


def test_the_filter_is_still_there():
    """M1 removes the controller and nothing else.

    An arbitrary command schedule, filtered by the equality rows, has to
    come out the same as filtering it with the filter. The filter is not
    the study's to change, and a bound that quietly dropped it would be
    reporting on a canal that does not exist.
    """
    programme, bound = model(0.7), free(0.7)
    rng = np.random.default_rng(20260910)
    commands = rng.normal(scale=0.2, size=(programme.steps, programme.response.size))
    x = vector(programme, np.zeros(programme.ratio.n_vars), commands)

    equality = np.abs(bound.ratio.a_eq @ x - bound.ratio.b_eq).max()
    assert equality < 1.0e-9, f"the filter and the plant rows disagree by {equality}"

    applied = bound.applied_of(x)
    assert np.abs(applied[:, 0] - apply_filter(SPEC, commands[:, 0])).max() < 1.0e-12
    assert np.abs(applied - commands).max() > 1.0e-3, (
        "the filter is passing the command through unchanged, so this test "
        "would pass with no filter at all"
    )


# ---------------------------------------------------------------------------
# What it is for
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_upper_bound_is_above_the_proposed_method():
    """The one end-to-end solve, and the reason it is marked slow.

    Everything above checks that the free-gate programme is the same
    canal, and all of it runs in seconds. This one actually solves it -
    five thousand variables, a lexicographic stage and a saturation test
    per user - and takes minutes. It is the number the paper quotes rather
    than the check that finds a mistake, so the working run leaves it out
    and a commit that touches the solvers or the plant puts it back with
    ``-m ""``.

    Equality is a legitimate outcome and is itself worth reporting - it
    means no control layer could have done better at that scarcity - so
    the test asserts the inequality and not a strict gap.
    """
    programme = model(0.7)
    best = leximin(programme)
    bound = upper_bound(programme, plant())

    assert bound.worst >= best.worst - best.detail["accuracy_bound"]
    assert bound.code == "M1"
    assert bound.detail["n_equalities"] == programme.steps * programme.response.size * 2
    assert bound.z.size == programme.ratio.n_vars


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_the_decision_vector_is_laid_out_where_it_says():
    programme, bound = model(0.7), free(0.7)
    grid = programme.steps * programme.response.size
    assert bound.offset_command == programme.ratio.n_vars
    assert bound.offset_applied == bound.offset_command + grid
    assert bound.offset_level == bound.offset_applied + grid
    assert bound.ratio.n_vars == bound.offset_level + grid

    x = np.arange(float(bound.ratio.n_vars))
    assert bound.orders_of(x)[0] == 0.0
    assert bound.commands_of(x).shape == (programme.steps, programme.response.size)
    assert bound.levels_of(x)[0, 0] == float(bound.offset_level)


def test_the_rows_that_mention_the_controller_are_the_only_ones_replaced():
    programme, bound = model(0.7), free(0.7)
    carried = sum(
        rows
        for name, rows in programme.row_counts.items()
        if name
        in {
            "C2 delivered flow non-negative",
            "C3 volume budget",
            "C8 source availability",
            "ratio capped at one",
        }
    )
    # C6 runs over every step now, not from the end of the warm-up: the
    # exemption belongs to the level band and to nothing else. The storage
    # state and its reconciliation band add four rows per pool and block,
    # counting the block a run starts from.
    travel = 2 * programme.response.size * (programme.steps - 1)
    storage = 4 * programme.response.size * (programme.response.blocks + 1)
    assert bound.ratio.a_ub.shape[0] == carried + travel + storage


def test_row_counts_that_do_not_add_up_are_refused():
    """The rows are taken by name, so the names have to account for them all."""
    from dataclasses import replace

    programme = model(0.7)
    broken = replace(programme, row_counts={"C3 volume budget": 8})
    with pytest.raises(UpperBoundError, match="row counts"):
        free_gate_programme(broken, plant())
