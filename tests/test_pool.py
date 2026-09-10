"""Permanent tests for the pool model. Never deleted.

The one that matters most is ``test_offtake_lowers_level``. The published
equation and the published figure caption disagree about the sign of the
offtake term, and the reading taken here - a withdrawal lowers the level -
is a decision, not a transcription. If anybody ever reverses it, every
result in the study inverts and nothing else would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircanal.benchmarks import HAUGHTON_POOLS, haughton_pool
from faircanal.pool import (
    PoolParams,
    homogeneous_roots,
    simulate_level,
    total_outflow,
)

ALL_POOLS = [
    (order, index)
    for order in sorted(HAUGHTON_POOLS)
    for index in sorted(HAUGHTON_POOLS[order])
]
POOL_IDS = [f"pool{index}-order{order}" for order, index in ALL_POOLS]


def zeros(steps: int) -> np.ndarray:
    return np.zeros(steps, dtype=float)


def step(steps: int, start: int = 0, height: float = 1.0) -> np.ndarray:
    signal = np.zeros(steps, dtype=float)
    signal[start:] = height
    return signal


# ---------------------------------------------------------------------------
# The sign convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_offtake_lowers_level(order: int, index: int):
    """A withdrawal takes water out of the pool, so the level must fall.

    With no inflow and no downstream outflow, a sustained offtake can only
    drain the reach. A positive final level would mean the sign of the
    offtake term has been reversed somewhere.
    """
    params = haughton_pool(index, order)
    steps = 400
    y = simulate_level(params, zeros(steps), zeros(steps), step(steps))
    assert y[-1] < 0.0, f"{params.name}: offtake raised the level to {y[-1]}"


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_inflow_raises_level(order: int, index: int):
    """The mirror image: water entering the reach must raise the level."""
    params = haughton_pool(index, order)
    steps = 400
    y = simulate_level(params, step(steps), zeros(steps), zeros(steps))
    assert y[-1] > 0.0, f"{params.name}: inflow lowered the level to {y[-1]}"


def test_offtake_matches_the_independently_computed_value():
    """Pin the level a documented scenario produces.

    Scenario: the more upstream Haughton reach, third-order model, no
    inflow and no downstream outflow, a unit offtake switched on at step
    100 and held, level read at step 399. An independent implementation of
    the same equation produced -18.0043 for this scenario, so the value is
    a cross-check between two implementations rather than a number this
    one certified about itself.

    The step starts at 100 rather than 0, and the value depends on that:
    the reach integrates, so the level keeps falling for as long as the
    offtake runs.
    """
    params = haughton_pool(1, 3)
    steps = 400
    y = simulate_level(params, zeros(steps), zeros(steps), step(steps, start=100))
    assert y[399] == pytest.approx(-18.0043, abs=5e-5)


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_outflow_and_offtake_are_the_same_sink(order: int, index: int):
    """Water leaving through the gate and water drawn by a user act alike.

    Both leave the reach at the point where the level is measured, so the
    model cannot tell them apart. If it could, one of the two signs is
    wrong.
    """
    params = haughton_pool(index, order)
    steps = 200
    through_gate = simulate_level(params, zeros(steps), step(steps), zeros(steps))
    through_offtake = simulate_level(params, zeros(steps), zeros(steps), step(steps))
    assert through_gate == pytest.approx(through_offtake, rel=0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_transport_delay_is_respected(order: int, index: int):
    """Inflow cannot be seen at the far end before it has travelled there."""
    params = haughton_pool(index, order)
    steps = 200
    y = simulate_level(params, step(steps), zeros(steps), zeros(steps))
    delay = params.tau + params.tau_bar
    assert y[: delay + 1] == pytest.approx(np.zeros(delay + 1), abs=1e-15), (
        f"{params.name}: the level moved before the inflow could arrive"
    )
    assert y[delay + 1] != 0.0, f"{params.name}: the inflow never arrived"


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_zero_input_stays_at_rest(order: int, index: int):
    """No flow anywhere means no movement, from any starting level."""
    params = haughton_pool(index, order)
    steps = 100
    y = simulate_level(params, zeros(steps), zeros(steps), zeros(steps), y_initial=0.37)
    assert y == pytest.approx(np.full(steps + 1, 0.37), abs=1e-12)


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_pool_is_an_integrator_with_a_damped_wave(order: int, index: int):
    """Exactly one root on the unit circle, the rest strictly inside it.

    That shape is what a canal reach is: storage that never forgets, plus
    a surface wave that dies away. A transcription error in the wave
    coefficients shows up here as a root outside the circle or as a second
    root on it.
    """
    params = haughton_pool(index, order)
    moduli = np.abs(homogeneous_roots(params))
    on_circle = np.isclose(moduli, 1.0, atol=1e-9)
    assert on_circle.sum() == 1, (
        f"{params.name}: expected one unit root, found moduli {moduli}"
    )
    assert np.all(moduli[~on_circle] < 1.0), (
        f"{params.name}: an unstable root, moduli {moduli}"
    )


def test_haughton_pool_1_root_moduli():
    """Pin the root moduli of the reach used in every structural check."""
    moduli = sorted(np.abs(homogeneous_roots(haughton_pool(1, 3))))
    assert moduli == pytest.approx([0.988939, 0.988939, 1.000000], abs=5e-7)


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_model_is_linear_in_the_flows(order: int, index: int):
    """Superposition holds, which is what makes the whole study a sequence
    of linear programmes rather than a non-convex search."""
    params = haughton_pool(index, order)
    steps = 150
    rng = np.random.default_rng(20260910)
    u1, o1, d1 = (rng.normal(size=steps) for _ in range(3))
    u2, o2, d2 = (rng.normal(size=steps) for _ in range(3))

    combined = simulate_level(params, u1 + 2.0 * u2, o1 + 2.0 * o2, d1 + 2.0 * d2)
    separate = simulate_level(params, u1, o1, d1) + 2.0 * simulate_level(
        params, u2, o2, d2
    )
    assert combined == pytest.approx(separate, abs=1e-9)


def test_branch_gates_enter_as_their_sum():
    """The tree extension: a parent reach sees the sum of its branches.

    This encodes the assumption that the branch gates sit at one cross
    section, so the parent's level does not depend on which branch the
    water leaves through. A cascade is the single-branch case, so the
    cascade and the tree agree by construction.
    """
    params = haughton_pool(1, 3)
    steps = 200
    rng = np.random.default_rng(11269)
    branch_a = rng.uniform(0.1, 1.0, size=steps)
    branch_b = rng.uniform(0.1, 1.0, size=steps)

    two_branches = simulate_level(
        params, zeros(steps), total_outflow([branch_a, branch_b]), zeros(steps)
    )
    one_equivalent_gate = simulate_level(
        params, zeros(steps), branch_a + branch_b, zeros(steps)
    )
    assert two_branches == pytest.approx(one_equivalent_gate, abs=1e-12)

    # A cascade is the single-branch case, unchanged.
    assert total_outflow([branch_a]) == pytest.approx(branch_a, abs=0.0)

    # And the second branch must actually reach the model.
    cascade = simulate_level(params, zeros(steps), branch_a, zeros(steps))
    assert not np.allclose(two_branches, cascade)


def test_a_leaf_pool_must_be_stated_explicitly():
    """An empty branch list is a mistake, not an implied zero.

    Silently returning zeros of an unknown length is how a leaf pool ends
    up with a horizon that does not match the rest of the network.
    """
    with pytest.raises(ValueError, match="leaf pool"):
        total_outflow([])


def test_branch_flows_must_share_a_horizon():
    with pytest.raises(ValueError, match="same shape"):
        total_outflow([zeros(10), zeros(11)])


# ---------------------------------------------------------------------------
# Refusals: a wrong parameter set must fail loudly, not quietly
# ---------------------------------------------------------------------------


def test_first_order_model_rejects_wave_coefficients():
    with pytest.raises(ValueError, match="no wave coefficients"):
        PoolParams(name="bad", b=(0.069,), c=(0.063,), alpha=(0.978, 0.468))


def test_third_order_model_rejects_the_filter_delay():
    """tau_bar belongs to the first-order approximation only."""
    with pytest.raises(ValueError, match="first-order"):
        PoolParams(
            name="bad",
            b=(0.137, 0.155, 0.053),
            c=(0.190, 0.333, 0.175),
            alpha=(0.978, 0.468),
            tau_bar=4,
        )


def test_mismatched_coefficient_counts_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        PoolParams(name="bad", b=(0.1, 0.2, 0.3), c=(0.1,), alpha=(0.5, 0.5))


def test_mismatched_signal_lengths_are_rejected():
    params = haughton_pool(1, 3)
    with pytest.raises(ValueError, match="same shape"):
        simulate_level(params, zeros(10), zeros(9), zeros(10))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("order", "index"), ALL_POOLS, ids=POOL_IDS)
def test_every_parameter_set_declares_where_it_came_from(order: int, index: int):
    """Every parameter is labelled observed, derived or assumed, with a
    source. A reviewer asks this of every number in a model, and the answer
    ships with the code rather than being reconstructed later."""
    params = haughton_pool(index, order)
    assert params.provenance in {"observed", "derived", "assumed"}
    assert params.source, f"{params.name}: no source recorded"
    assert params.units, f"{params.name}: no unit recorded"
