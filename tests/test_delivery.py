"""Permanent tests for the filter and the delivery operator. Never deleted.

Three of these carry the study's central claim. The filter conserves
volume over an unbounded horizon, it does not conserve volume inside a
delivery window, and it drives a rectangular order below zero. Together
those say that a farmer cannot receive the ordered profile no matter how
the network is operated, which is the gap this study closes.

The pinned fractions are cross-checks against an independently written
script that filtered the same rectangular order, not values this code
certified about itself. They are pinned to nine decimals rather than to
bit equality, because the filter coefficients are computed in floating
point and the last two digits move between library versions. Nine decimals
is far tighter than anything the study reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircanal.benchmarks import HAUGHTON_FILTER, haughton_filter
from faircanal.delivery import (
    FilterSpec,
    apply_filter,
    dc_gain,
    delivery_operator,
    filter_matrix,
    hold_matrix,
    impulse_response,
    memory_steps,
)

ORDERS = (
    HAUGHTON_FILTER["order_in_paper"],
    HAUGHTON_FILTER["order_in_released_code"],
)

#: A rectangular order 201 steps long, filtered over a 20 000 step horizon.
#: Independently computed reference values, one entry per filter order.
PULSE_STEPS = 201
LONG_HORIZON = 20_000
REFERENCE = {
    3: {"in_window": 0.9448701750371497, "long_horizon": 1.0, "minimum": -0.08232927962051315},
    4: {"in_window": 0.9279694297897777, "long_horizon": 1.0, "minimum": -0.10901707189805404},
}

#: Filter memory in plant steps: the number of leading impulse-response
#: samples that exceed 1e-4 in absolute value. The last index above the
#: threshold is one less than this, which is the number an earlier note
#: recorded; both readings sit well inside the horizon margin.
MEMORY_STEPS = {3: 82, 4: 103}


def pulse_response(order: int) -> np.ndarray:
    x = np.zeros(LONG_HORIZON, dtype=float)
    x[:PULSE_STEPS] = 1.0
    return apply_filter(haughton_filter(order), x)


# ---------------------------------------------------------------------------
# What the filter preserves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_dc_gain_is_one(order: int):
    """Unit steady-state gain: over an unbounded horizon, no water is lost."""
    assert dc_gain(haughton_filter(order)) == pytest.approx(1.0, abs=1e-11)


@pytest.mark.parametrize("order", ORDERS)
def test_volume_is_conserved_over_a_long_horizon(order: int):
    """Every cubic metre released is eventually delivered.

    This is what makes the study's measure honest: the method cannot make
    a user better off by releasing more water, only by releasing it at a
    different time, so a comparison at equal released volume is meaningful.
    """
    delivered = pulse_response(order).sum() / PULSE_STEPS
    assert delivered == pytest.approx(REFERENCE[order]["long_horizon"], abs=1e-9)


@pytest.mark.parametrize("order", ORDERS)
def test_normalised_cutoff_is_relative_to_nyquist(order: int):
    """Cut-off normalised by the Nyquist frequency pi / T, not by 1 / T.

    A factor of two here changes every number in the study and nothing
    downstream would notice, so it is derived and checked rather than
    written out once.
    """
    spec = haughton_filter(order)
    expected = spec.cutoff_rad_per_s * spec.sample_time_s / np.pi
    assert spec.normalised_cutoff == pytest.approx(expected, rel=1e-15)
    assert spec.normalised_cutoff == pytest.approx(0.0572957795130823, abs=1e-15)


# ---------------------------------------------------------------------------
# What the filter costs the user - the premise of the whole study
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_part_of_the_order_arrives_after_the_window_closes(order: int):
    """Inside a finite window the filter does not conserve volume.

    The impulse response has an infinite tail, so an order that is not
    shifted forward in time always delivers less than it released while
    the window is open. Unshifted delivery is therefore strictly short of
    the order even with unlimited capacity, no losses and a single reach.
    """
    fraction = pulse_response(order)[:PULSE_STEPS].sum() / PULSE_STEPS
    assert fraction == pytest.approx(REFERENCE[order]["in_window"], abs=1e-9)
    assert fraction < 1.0


@pytest.mark.parametrize("order", ORDERS)
def test_a_rectangular_order_is_driven_below_zero(order: int):
    """The filtered order goes negative, and a gate cannot do that.

    This is why non-negativity of the delivered discharge is a constraint
    of the optimisation rather than something that holds automatically:
    reproducing the ordered profile exactly would require the offtake to
    run backwards. The requirement is measured here, not assumed.
    """
    minimum = pulse_response(order).min()
    assert minimum == pytest.approx(REFERENCE[order]["minimum"], abs=1e-9)
    assert minimum < 0.0


@pytest.mark.parametrize("order", ORDERS)
def test_a_higher_order_filter_costs_more(order: int):
    """Sanity of the ordering, so a swapped pair of numbers is caught."""
    third = pulse_response(3)
    fourth = pulse_response(4)
    assert fourth[:PULSE_STEPS].sum() < third[:PULSE_STEPS].sum()
    assert fourth.min() < third.min()


# ---------------------------------------------------------------------------
# Memory and the horizon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
def test_filter_memory(order: int):
    assert memory_steps(haughton_filter(order)) == MEMORY_STEPS[order]


def test_memory_refuses_to_guess_when_it_has_not_settled():
    """A horizon that is too short is an error, not a smaller answer."""
    with pytest.raises(RuntimeError, match="had not settled"):
        memory_steps(haughton_filter(3), max_steps=20)


# ---------------------------------------------------------------------------
# The delivery operator
# ---------------------------------------------------------------------------


def test_delivery_operator_equals_g_times_z():
    """The cheap construction agrees with the definition."""
    spec = haughton_filter(3)
    steps, blocks, per_block = 300, 20, 15
    gamma = delivery_operator(spec, steps, blocks, per_block)
    explicit = filter_matrix(spec, steps) @ hold_matrix(steps, blocks, per_block)
    assert gamma == pytest.approx(explicit, abs=1e-12)


def test_delivery_operator_is_causal():
    """An order placed in block j cannot deliver water before block j starts."""
    spec = haughton_filter(3)
    steps, blocks, per_block = 300, 20, 15
    gamma = delivery_operator(spec, steps, blocks, per_block)
    for j in range(blocks):
        head = gamma[: j * per_block, j]
        assert head == pytest.approx(np.zeros(head.size), abs=1e-15), (
            f"block {j} delivered water before it began"
        )


def test_each_block_delivers_its_whole_volume_eventually():
    """A column of the operator carries the volume of one full block.

    With unit DC gain, a unit order held for ``per_block`` steps delivers
    ``per_block`` step-units of volume, provided the horizon is long enough
    to hold the tail.
    """
    spec = haughton_filter(4)
    per_block = 15
    blocks = 4
    steps = blocks * per_block + 20 * MEMORY_STEPS[4]
    gamma = delivery_operator(spec, steps, blocks, per_block)
    assert gamma.sum(axis=0) == pytest.approx(np.full(blocks, per_block), abs=1e-6)


def test_delivery_is_linear_in_the_order():
    """Superposition: this is why the problem is a linear programme."""
    spec = haughton_filter(3)
    steps, blocks, per_block = 400, 20, 15
    gamma = delivery_operator(spec, steps, blocks, per_block)
    rng = np.random.default_rng(2203)
    v1 = rng.uniform(0.0, 2.0, size=blocks)
    v2 = rng.uniform(0.0, 2.0, size=blocks)
    assert gamma @ (v1 + 3.0 * v2) == pytest.approx(
        gamma @ v1 + 3.0 * (gamma @ v2), abs=1e-9
    )


def test_filtering_a_block_order_matches_the_operator():
    """The operator is the filter, not an approximation of it."""
    spec = haughton_filter(3)
    steps, blocks, per_block = 400, 20, 15
    rng = np.random.default_rng(16575)
    v = rng.uniform(0.0, 2.0, size=blocks)
    held = hold_matrix(steps, blocks, per_block) @ v
    assert delivery_operator(spec, steps, blocks, per_block) @ v == pytest.approx(
        apply_filter(spec, held), abs=1e-12
    )


# ---------------------------------------------------------------------------
# Structure of the hold, and refusals
# ---------------------------------------------------------------------------


def test_hold_matrix_structure():
    Z = hold_matrix(steps=100, blocks=4, steps_per_block=15)
    assert Z.shape == (100, 4)
    assert Z.sum() == 60.0
    assert (Z.sum(axis=1) <= 1.0).all(), "a plant step belongs to at most one block"
    assert Z[59, 3] == 1.0 and Z[60:, :].sum() == 0.0


def test_hold_matrix_rejects_a_horizon_that_cannot_hold_the_blocks():
    with pytest.raises(ValueError, match="does not fit"):
        hold_matrix(steps=50, blocks=4, steps_per_block=15)


def test_filter_family_is_not_substituted():
    with pytest.raises(ValueError, match="Butterworth"):
        FilterSpec(
            order=3, cutoff_rad_per_s=3.0e-3, sample_time_s=60.0, family="chebyshev"
        )


def test_cutoff_at_or_above_nyquist_is_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        FilterSpec(order=3, cutoff_rad_per_s=1.0, sample_time_s=60.0)


def test_filter_order_outside_the_published_pair_is_refused():
    """Neither three nor four is a default, and five is not an option."""
    with pytest.raises(ValueError, match="refusing to build"):
        haughton_filter(5)


def test_impulse_response_starts_from_rest():
    """No output before the impulse, and a response afterwards."""
    h = impulse_response(haughton_filter(3), 200)
    assert h[0] != 0.0
    assert np.abs(h).max() > 0.0


def test_filter_is_causal():
    """Nothing comes out before something goes in."""
    spec = haughton_filter(3)
    x = np.zeros(300, dtype=float)
    x[120:] = 1.0
    y = apply_filter(spec, x)
    assert y[:120] == pytest.approx(np.zeros(120), abs=1e-15)
    assert y[120] != 0.0


def test_the_filter_order_is_the_one_the_paper_states():
    """Three, from Section 4.1, not four from the released code.

    The paper says "the final design is a third-order filter with a
    cut-off frequency 3e-3 rad/sec"; the authors' code uses four. That is a
    difference between their paper and their code, and this study follows
    the paper. The fourth order stays reachable so the choice can be shown
    not to drive the results, but it is not the default and there is no
    default.
    """
    assert HAUGHTON_FILTER["order_used"] == HAUGHTON_FILTER["order_in_paper"] == 3
    assert HAUGHTON_FILTER["order_in_released_code"] == 4
    assert haughton_filter(HAUGHTON_FILTER["order_used"]).order == 3
