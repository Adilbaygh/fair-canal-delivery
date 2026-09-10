"""Permanent tests for the closed-loop response map. Never deleted.

The map is the canal as the linear programme sees it, so an error here
does not produce a wrong number - it produces a different canal, and every
result after it is internally consistent and about that other canal. Two
of these tests exist for one mistake each: building the map from the
feedback loop alone, and cutting the horizon before the filter's tail
arrives. Both leave a map that behaves and is wrong.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from faircanal.benchmarks import haughton_filter
from faircanal.closedloop import simulate_closed_loop
from faircanal.config import SETTLE_MARGIN_STEPS, STEPS_PER_BLOCK
from faircanal.control import control_law
from faircanal.delivery import hold_matrix, memory_steps
from faircanal.network import corning_cascade, corning_tree
from faircanal.plant import (
    CanalPlant,
    PlantError,
    build_plant,
    feedforward_free_law,
    horizon_for,
    response_map,
)

SPEC = haughton_filter(3)
BLOCKS = 4


@lru_cache(maxsize=4)
def plant():
    return build_plant(corning_cascade(), SPEC)


@lru_cache(maxsize=4)
def law(steps: int):
    return control_law(list(plant().design_models), plant().weights, horizon=steps)


@lru_cache(maxsize=4)
def mapping(blocks: int = BLOCKS):
    steps = horizon_for(blocks)
    return response_map(plant(), blocks, law=law(steps))


# ---------------------------------------------------------------------------
# The map is the loop
# ---------------------------------------------------------------------------


def test_the_map_reproduces_a_direct_simulation():
    """A matrix product against the loop it was taken from.

    The loop is linear, so the map is not an approximation of it - it is
    it. Anything above rounding here means a column was collected wrongly,
    and no other test in the suite would notice.
    """
    reference = mapping()
    orders = np.random.default_rng(3).uniform(0.0, 0.4, size=(reference.size, BLOCKS))
    hold = hold_matrix(reference.steps, BLOCKS, reference.steps_per_block)
    direct = simulate_closed_loop(
        plant().loop,
        np.zeros(reference.size),
        hold @ orders.T,
        law=law(reference.steps),
    )
    assert reference.levels_at(orders) == pytest.approx(
        direct.levels[: reference.steps], abs=1e-11
    )
    assert reference.commands_at(orders) == pytest.approx(direct.commanded, abs=1e-10)
    assert reference.delivered_at(orders) == pytest.approx(
        direct.offtake_applied, abs=1e-11
    )


def test_the_map_must_be_built_as_one_whole():
    """An order reaches the canal by two roads; a map built from one is wrong.

    The source's controller has no integral action and rejects load
    disturbances entirely by feed-forward. Building the map with the
    feed-forward paths removed - the mistake the module docstring
    describes - leaves a map that is still linear, still reproduces its own
    loop, and answers about a different system. It has to disagree, and by
    a lot.
    """
    whole = mapping()
    crippled = response_map(
        plant(), BLOCKS, law=feedforward_free_law(law(whole.steps))
    )
    gap = np.abs(crippled.commands - whole.commands).max() / np.abs(whole.commands).max()
    assert gap > 0.5, (
        f"dropping the feed-forward now changes the map by only {gap:.1%}, so this "
        f"test no longer protects anything"
    )
    level_gap = np.abs(crippled.levels - whole.levels).max() / np.abs(whole.levels).max()
    assert level_gap > 0.5


def test_the_map_is_linear():
    """Scaling and superposition, because everything downstream assumes them."""
    reference = mapping()
    rng = np.random.default_rng(17)
    first = rng.uniform(0.0, 0.3, size=(reference.size, BLOCKS))
    second = rng.uniform(0.0, 0.3, size=(reference.size, BLOCKS))
    assert reference.levels_at(2.5 * first) == pytest.approx(
        2.5 * reference.levels_at(first), abs=1e-12
    )
    assert reference.levels_at(first + second) == pytest.approx(
        reference.levels_at(first) + reference.levels_at(second), abs=1e-12
    )


def test_the_baseline_is_nothing_when_the_canal_starts_at_its_set_point():
    reference = mapping()
    assert np.abs(reference.baseline_levels).max() == 0.0
    assert np.abs(reference.baseline_commands).max() == 0.0
    assert reference.levels_at(np.zeros((reference.size, BLOCKS))).max() == 0.0


def test_a_starting_offset_lands_in_the_baseline_and_not_in_the_map():
    """Where the initial condition goes, and that it goes only there.

    The response to the orders and the response to the initial state are
    two separate terms of the same affine map, and mixing them would make
    every column depend on where the canal happened to start.
    """
    steps = horizon_for(BLOCKS)
    offset = np.full(plant().size, 0.05)
    shifted = response_map(plant(), BLOCKS, initial_levels=offset, law=law(steps))
    assert np.abs(shifted.baseline_levels).max() > 1e-6
    assert shifted.levels == pytest.approx(mapping().levels, abs=1e-10)
    assert shifted.commands == pytest.approx(mapping().commands, abs=1e-9)


# ---------------------------------------------------------------------------
# The delivered water
# ---------------------------------------------------------------------------


def test_the_delivered_flow_is_the_filter_after_the_hold():
    """Every block's order arrives in full, late and smeared but whole.

    The filter has unit gain, so a block of order carries a block's worth
    of water - as long as the run is long enough for its tail. That is the
    volume balance the whole fairness measure rests on.
    """
    reference = mapping()
    volumes = reference.delivery.sum(axis=0)
    assert volumes == pytest.approx(
        np.full(BLOCKS, float(reference.steps_per_block)), rel=1e-4
    )


def test_the_delivered_flow_is_the_same_at_every_gate():
    """One filter, so one delivery operator, whatever gate a user sits on."""
    reference = mapping()
    orders = np.zeros((reference.size, BLOCKS))
    orders[0, 1] = 1.0
    first = reference.delivered_at(orders)[:, 0]
    orders = np.zeros((reference.size, BLOCKS))
    orders[reference.size - 1, 1] = 1.0
    last = reference.delivered_at(orders)[:, reference.size - 1]
    assert first == pytest.approx(last, abs=1e-15)


# ---------------------------------------------------------------------------
# The horizon
# ---------------------------------------------------------------------------


def test_the_horizon_is_the_ordering_window_plus_the_settling_margin():
    assert horizon_for(1) == STEPS_PER_BLOCK + SETTLE_MARGIN_STEPS
    assert horizon_for(6) == 6 * STEPS_PER_BLOCK + SETTLE_MARGIN_STEPS
    with pytest.raises(PlantError, match="at least one"):
        horizon_for(0)


def test_a_margin_shorter_than_the_filter_is_refused():
    """The quiet failure: the last block's tail falls off the end.

    Nothing complains, the map is still linear, and the volume balance is
    silently short. So the margin is checked against the filter it has to
    cover rather than trusted.
    """
    memory = memory_steps(SPEC)
    assert SETTLE_MARGIN_STEPS >= memory
    with pytest.raises(PlantError, match="shorter than the filter"):
        response_map(plant(), 2, margin=memory - 1)


def test_the_law_and_the_run_must_agree_on_the_horizon():
    with pytest.raises(PlantError, match="horizon"):
        response_map(plant(), BLOCKS, law=law(horizon_for(BLOCKS + 1)))


# ---------------------------------------------------------------------------
# What the map says about this canal
# ---------------------------------------------------------------------------


def test_the_upstream_gates_move_further_than_the_gate_that_ordered():
    """A property of the source's weights on a canal of this size.

    Its Section 2.2 requires no cost on any gate but the reservoir, so the
    internal gates are free to move and the optimum uses them. On a canal
    whose pools store as much as this one, a level barely moves for a large
    flow, so the flows come out large: ordering one unit at the far end
    swings a middle gate by more than eight.

    That is not a fault to fix here - the weight structure is the method's
    own condition - but it decides whether the capacity constraint binds,
    so it is measured and reported rather than discovered later.
    """
    gains = mapping().peak_command_gain()
    assert gains.shape == (mapping().size, mapping().size)
    assert gains.max() > 5.0
    ordered_at_the_end = gains[:, 0]
    assert ordered_at_the_end[0] > 4.0, (
        f"one unit ordered at the far gate now swings that gate by only "
        f"{ordered_at_the_end[0]:.2f}"
    )
    assert ordered_at_the_end[2] > ordered_at_the_end[0], (
        "the gate that ordered now moves more than the ones upstream of it, "
        "so the amplification has gone"
    )
    # The reservoir gate is the one that carries a cost, and it barely moves.
    assert gains[mapping().size - 1].max() < 0.05


def test_the_largest_command_is_the_start_up_transient():
    """Where the biggest number in the map is, and that it is one step wide.

    At the first step the controller learns the whole future at once and
    pre-positions for all of it. Everything after that is half the size.
    The model document provides for this as an initial transient, and the
    programme handles it there rather than here; recording it makes sure
    nobody later reads the peak as a steady requirement.
    """
    reference = mapping()
    shaped = reference.commands.reshape(
        reference.steps, reference.size, reference.size, reference.blocks
    )
    first = float(np.abs(shaped[0]).max())
    rest = float(np.abs(shaped[1:]).max())
    assert first > rest
    assert first / rest > 1.4
    assert rest > 3.0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_branched_network_is_refused():
    """The controller assembled here is a cascade controller, and says so."""
    tree = corning_tree()
    with pytest.raises(PlantError, match="cascade controller"):
        build_plant(tree, SPEC)


def test_a_plant_with_the_wrong_number_of_models_is_refused():
    base = plant()
    with pytest.raises(PlantError, match="reaches but"):
        CanalPlant(
            network=base.network,
            plant_models=base.plant_models[:3],
            design_models=base.design_models,
            filter_spec=SPEC,
            weights=base.weights,
            fit=base.fit,
        )


def test_the_column_index_is_node_major():
    reference = mapping()
    assert reference.column(1, 0) == 0
    assert reference.column(1, BLOCKS - 1) == BLOCKS - 1
    assert reference.column(2, 0) == BLOCKS
    assert reference.column(reference.size, BLOCKS - 1) == reference.size * BLOCKS - 1
    with pytest.raises(PlantError, match="outside"):
        reference.column(0, 0)
    with pytest.raises(PlantError, match="outside"):
        reference.column(1, BLOCKS)


def test_orders_of_the_wrong_shape_are_refused():
    reference = mapping()
    with pytest.raises(PlantError, match="shape"):
        reference.levels_at(np.zeros((reference.size, BLOCKS + 1)))
    with pytest.raises(PlantError, match="shape"):
        reference.delivered_at(np.zeros(BLOCKS))


def test_a_starting_level_of_the_wrong_shape_is_refused():
    with pytest.raises(PlantError, match="initial_levels"):
        response_map(plant(), 2, initial_levels=np.zeros(3))
