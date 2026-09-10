"""The closed loop as one linear map, built by running it whole.

What this module produces
-------------------------
Everything downstream of here is a linear programme, and a linear
programme needs the canal as a matrix. This module produces that matrix:
how the level of every pool and the command at every gate move when a
block of water is ordered at one gate.

Because the loop is linear - a linear plant, a linear filter, a linear
observer and a linear control law - the map exists and is exact. It is not
an approximation of the loop; it *is* the loop, written down.

The trap this module exists to avoid
------------------------------------
An order reaches the canal by two roads, not one:

1. through the plant, as water actually drawn at a gate, after the filter;
2. through the controller, as an *announced* disturbance that the
   feed-forward acts on before the water moves.

The source's controller has no integral action and leans entirely on the
second road - "as the structured controller does not have integral action
but instead relies on feed-forward to reject load disturbances". A map
assembled from the feedback loop alone loses that road. The loop stays
linear, the tests still pass, and the answer belongs to a different
system - one where reshaping an order cannot pre-position the gates, which
is precisely the mechanism this study is about.

So the map is built by driving the whole loop: one unit order on one gate
in one block, with the controller told about it and the plant receiving
its filtered version, both from the same call.
``test_the_map_must_be_built_as_one_whole`` builds the crippled version
too and requires it to disagree.

Why one column per gate and not per user
----------------------------------------
Users at the same gate order into the same channel, and the filter and the
loop treat them identically. So the loop is driven once per (gate, block)
and the users are combined afterwards, which is what makes the map cost
``N x B`` simulations rather than one per user.

Sign convention
---------------
Offtakes are physical throughout: a positive order takes water out of the
canal. The controller's own convention is the opposite and
:func:`faircanal.closedloop.simulate_closed_loop` does that flip
internally, so nothing here has to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faircanal.closedloop import ClosedLoop, simulate_closed_loop
from faircanal.config import DT_PLANT_S, SETTLE_MARGIN_STEPS, STEPS_PER_BLOCK
from faircanal.control import LinearControlLaw, LqWeights, control_law
from faircanal.delivery import FilterSpec, delivery_operator, hold_matrix, memory_steps
from faircanal.design import DelayFit, design_models
from faircanal.network import Network, NetworkError, identify_network
from faircanal.pool import PoolParams

__all__ = [
    "PlantError",
    "CanalPlant",
    "ResponseMap",
    "build_plant",
    "horizon_for",
    "response_map",
    "feedforward_free_law",
]


class PlantError(ValueError):
    """Raised when a plant or a response map cannot be assembled."""


@dataclass(frozen=True)
class CanalPlant:
    """A canal, its two models of itself, its filter and its controller.

    ``plant_models`` are the third-order models the response is evaluated
    on; ``design_models`` are the first-order ones the controller was
    designed against. Keeping them apart is the whole reason the loop has
    an observer, and mixing them would quietly remove the mismatch this
    study runs on.
    """

    network: Network
    plant_models: tuple[PoolParams, ...]
    design_models: tuple[PoolParams, ...]
    filter_spec: FilterSpec
    weights: LqWeights
    fit: DelayFit
    kalman_r1: float = 1.0
    kalman_r2: float = 100.0
    kalman_reading: str = "covariance"

    def __post_init__(self) -> None:
        size = self.network.size
        if len(self.plant_models) != size or len(self.design_models) != size:
            raise PlantError(
                f"{self.network.name}: {size} reaches but "
                f"{len(self.plant_models)} plant models and "
                f"{len(self.design_models)} design models"
            )
        if not self.network.is_cascade:
            raise PlantError(
                f"{self.network.name}: the controller of the source study is a "
                f"cascade controller, so a branched network needs a different "
                f"synthesis than the one this plant assembles"
            )

    @property
    def size(self) -> int:
        return self.network.size

    @property
    def loop(self) -> ClosedLoop:
        return ClosedLoop(
            plant_pools=self.plant_models,
            design_pools=self.design_models,
            filter_spec=self.filter_spec,
            weights=self.weights,
            kalman_r1=self.kalman_r1,
            kalman_r2=self.kalman_r2,
            kalman_reading=self.kalman_reading,
        )


def build_plant(
    network: Network,
    filter_spec: FilterSpec,
    weights: LqWeights | None = None,
    dt: float = DT_PLANT_S,
) -> CanalPlant:
    """Identify the reaches, fit the design models, and wire the loop.

    The weights default to the source's own, which its Section 2.2 states
    as a condition of the method rather than a preference: ``r_i = 0``
    everywhere except the reservoir gate, and no rate term.
    """
    if weights is None:
        weights = LqWeights()
    plant_models = identify_network(network, dt=dt)
    models, fit = design_models(plant_models, filter_spec)
    return CanalPlant(
        network=network,
        plant_models=plant_models,
        design_models=models,
        filter_spec=filter_spec,
        weights=weights,
        fit=fit,
    )


def horizon_for(
    blocks: int,
    steps_per_block: int = STEPS_PER_BLOCK,
    margin: int = SETTLE_MARGIN_STEPS,
) -> int:
    """How many plant steps a run of *blocks* order blocks needs.

    The ordering window plus enough afterwards for the filter's tail to
    arrive. Cutting it short would lose part of the last block's water and
    the volume balance would fail quietly, which is why the margin is a
    setting and not a guess - see
    ``tests/test_config.py::test_settle_margin_covers_the_filter_memory``.
    """
    if blocks < 1:
        raise PlantError("a run needs at least one order block")
    return blocks * steps_per_block + margin


def feedforward_free_law(law: LinearControlLaw) -> LinearControlLaw:
    """The same law with both feed-forward paths removed.

    Exists so that the mistake the module docstring describes can be built
    on purpose and shown to give a different answer. Nothing else uses it.
    """
    return LinearControlLaw(
        k_levels=law.k_levels,
        k_history=law.k_history,
        k_preview=np.zeros_like(law.k_preview),
        k_disturbance_history=np.zeros_like(law.k_disturbance_history),
        lag=law.lag,
        horizon=law.horizon,
    )


@dataclass(frozen=True)
class ResponseMap:
    """The loop as a matrix, plus what it does when nobody orders anything.

    ``levels`` and ``commands`` each have one column per ``(node, block)``
    pair, ordered node-major: column ``(n - 1) * blocks + j``. Their rows
    run over ``(step, node)`` in the same order
    :func:`faircanal.closedloop.simulate_closed_loop` returns them,
    flattened.

    ``delivery`` is the same ``Gamma`` every gate uses - filter after
    hold - so a user's delivered flow is ``delivery @ v_i`` whatever gate
    they sit on.
    """

    levels: np.ndarray
    commands: np.ndarray
    baseline_levels: np.ndarray
    baseline_commands: np.ndarray
    delivery: np.ndarray
    steps: int
    blocks: int
    steps_per_block: int
    size: int

    def __post_init__(self) -> None:
        columns = self.size * self.blocks
        rows = self.steps * self.size
        for name, matrix, shape in (
            ("levels", self.levels, (rows, columns)),
            ("commands", self.commands, (rows, columns)),
        ):
            if matrix.shape != shape:
                raise PlantError(f"{name} has shape {matrix.shape}, expected {shape}")
        if self.delivery.shape != (self.steps, self.blocks):
            raise PlantError(
                f"delivery has shape {self.delivery.shape}, expected "
                f"{(self.steps, self.blocks)}"
            )

    def column(self, node: int, block: int) -> int:
        """Where one gate's one block sits in the map."""
        if not 1 <= node <= self.size:
            raise PlantError(f"node {node} is outside 1..{self.size}")
        if not 0 <= block < self.blocks:
            raise PlantError(f"block {block} is outside 0..{self.blocks - 1}")
        return (node - 1) * self.blocks + block

    def flatten_orders(self, orders: np.ndarray) -> np.ndarray:
        """Turn a ``(size, blocks)`` table of orders into the map's vector."""
        orders = np.asarray(orders, dtype=float)
        if orders.shape != (self.size, self.blocks):
            raise PlantError(
                f"orders must have shape {(self.size, self.blocks)}, got "
                f"{orders.shape}"
            )
        return orders.reshape(-1)

    def levels_at(self, orders: np.ndarray) -> np.ndarray:
        """Level deviations for a whole table of orders, ``(steps, size)``."""
        value = self.baseline_levels + self.levels @ self.flatten_orders(orders)
        return value.reshape(self.steps, self.size)

    def commands_at(self, orders: np.ndarray) -> np.ndarray:
        """Commanded gate flows for a whole table of orders."""
        value = self.baseline_commands + self.commands @ self.flatten_orders(orders)
        return value.reshape(self.steps, self.size)

    def peak_command_gain(self) -> np.ndarray:
        """Largest gate movement per unit order, ``[gate, gate ordered]``.

        How far each gate has to swing when one unit is ordered at another,
        taken over the whole run and every block. Reported because it is
        what decides whether the capacity constraint binds, and because on
        a large canal it is much bigger than one - see
        ``test_the_upstream_gates_move_further_than_the_gate_that_ordered``.
        """
        shaped = self.commands.reshape(
            self.steps, self.size, self.size, self.blocks
        )
        return np.abs(shaped).max(axis=(0, 3))

    def delivered_at(self, orders: np.ndarray) -> np.ndarray:
        """Delivered flow at each gate, ``(steps, size)``.

        The same filter acts on every gate, so this is one matrix product
        and needs no simulation.
        """
        orders = np.asarray(orders, dtype=float)
        if orders.shape != (self.size, self.blocks):
            raise PlantError(
                f"orders must have shape {(self.size, self.blocks)}, got "
                f"{orders.shape}"
            )
        return (self.delivery @ orders.T).reshape(self.steps, self.size)


def response_map(
    plant: CanalPlant,
    blocks: int,
    initial_levels: np.ndarray | None = None,
    steps_per_block: int = STEPS_PER_BLOCK,
    margin: int = SETTLE_MARGIN_STEPS,
    law: LinearControlLaw | None = None,
) -> ResponseMap:
    """Drive the whole loop once per gate and block, and collect the columns.

    Parameters
    ----------
    plant:
        The assembled canal.
    blocks:
        How many order blocks the decision covers.
    initial_levels:
        Level deviation at the start. Defaults to the set-point, which is
        the steady operating point the model document requires; anything
        else shows up in the baseline rather than in the map.
    law:
        A previously extracted control law, to save rebuilding it.

    The horizon is the ordering window plus the settling margin, so the
    filter's tail from the last block lands inside the run. Every column
    is the whole loop's answer to one unit order: the controller is told
    about it and the plant receives its filtered version, from one call.
    """
    size = plant.size
    steps = horizon_for(blocks, steps_per_block, margin)
    if initial_levels is None:
        initial_levels = np.zeros(size)
    initial_levels = np.asarray(initial_levels, dtype=float)
    if initial_levels.shape != (size,):
        raise PlantError(f"initial_levels must have shape ({size},)")

    loop = plant.loop
    if law is None:
        law = control_law(list(plant.design_models), plant.weights, horizon=steps)
    if law.horizon != steps:
        raise PlantError(
            f"the law was built for a horizon of {law.horizon}, but this run needs "
            f"{steps}"
        )

    memory = memory_steps(plant.filter_spec)
    if margin < memory:
        raise PlantError(
            f"the settling margin of {margin} steps is shorter than the filter's "
            f"memory of {memory}; the last block's water would fall outside the run"
        )

    baseline = simulate_closed_loop(
        loop, initial_levels, np.zeros((steps, size)), law=law
    )
    baseline_levels = baseline.levels[:steps].reshape(-1).copy()
    baseline_commands = baseline.commanded.reshape(-1).copy()

    hold = hold_matrix(steps, blocks, steps_per_block)
    columns_levels = np.zeros((steps * size, size * blocks))
    columns_commands = np.zeros((steps * size, size * blocks))

    for node in range(1, size + 1):
        for block in range(blocks):
            offtake = np.zeros((steps, size))
            offtake[:, node - 1] = hold[:, block]
            run = simulate_closed_loop(loop, initial_levels, offtake, law=law)
            index = (node - 1) * blocks + block
            columns_levels[:, index] = (
                run.levels[:steps].reshape(-1) - baseline_levels
            )
            columns_commands[:, index] = (
                run.commanded.reshape(-1) - baseline_commands
            )

    return ResponseMap(
        levels=columns_levels,
        commands=columns_commands,
        baseline_levels=baseline_levels,
        baseline_commands=baseline_commands,
        delivery=delivery_operator(
            plant.filter_spec, steps, blocks, steps_per_block
        ),
        steps=steps,
        blocks=blocks,
        steps_per_block=steps_per_block,
        size=size,
    )
