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

Two gate signals, not one
-------------------------
The controller asks for a flow and the filter decides what the gate
actually passes, so ``commanded`` and ``applied`` are different signals -
the gap between them is the filter this study exists to work around. Both
are carried. ``applied`` is the flow through the gate, which is what the
conveyance limit and the travel rate are limits on and what the model
document's ``u_n[k]`` means; ``commanded`` is kept because the free-gate
bound frees exactly that signal and because a test needs to show the two
differ rather than assume it.

Which one a constraint belongs on is not a matter of taste. A third-order
Butterworth overshoots, so the applied flow can exceed the commanded one
on a step, and a capacity row written on the command would be satisfied by
a schedule the canal cannot pass.

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
from faircanal.config import (
    BAND_AREA,
    DT_PLANT_S,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.control import LinearControlLaw, LqWeights, control_law
from faircanal.delivery import FilterSpec, delivery_operator, hold_matrix, memory_steps
from faircanal.design import DelayFit, design_models
from faircanal.network import Network, NetworkError, identify_network, surface_area
from faircanal.pool import PoolParams

__all__ = [
    "PlantError",
    "CanalPlant",
    "ResponseMap",
    "build_plant",
    "horizon_for",
    "response_map",
    "feedforward_free_law",
    "backwater_areas",
    "storage_areas",
    "control_signature",
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


def backwater_areas(models) -> tuple[float, ...]:
    """The area each identified pool integrates its net flow at [m^2].

    Not the water surface. The third-order model's integrator gain is
    ``(b1 - b2 + b3) / (1 - alpha_2)`` metres per cubic metre per second
    per step, and that gain is ``dt / A_d`` by construction, so the
    backwater area comes straight back out of the coefficients that were
    fitted. On the canal of this study it is between 13 and 76 per cent
    below the surface area, which is why C9' uses this one: a band written
    on a quantity the plant does not integrate at drifts with the flow.
    """
    return tuple(
        DT_PLANT_S * (1.0 - pool.alpha[1]) / (pool.b[0] - pool.b[1] + pool.b[2])
        for pool in models
    )


def storage_areas(plant: CanalPlant, choice: str = BAND_AREA) -> tuple[float, ...]:
    """The area C9' converts a level into a storage with [m^2]."""
    if choice == "backwater":
        return backwater_areas(plant.plant_models)
    if choice == "surface":
        return tuple(surface_area(reach) for reach in plant.network.reaches)
    raise PlantError(
        f"the storage area must be 'backwater' or 'surface', not {choice!r}"
    )


def control_signature(plant: CanalPlant) -> dict:
    """Everything about the loop that an answer depends on.

    Carried into the certificate's digest so that two runs differing in
    the filter or in the control law cannot share a digest. They used to:
    the payload named the scenario and nothing else, so the sensitivity
    runs collided with the main scan and the article reported the
    collision with the wrong reason attached.
    """
    return {
        "filter_order": int(plant.filter_spec.order),
        "filter_cutoff_rad_per_s": float(plant.filter_spec.cutoff_rad_per_s),
        "filter_sample_time_s": float(plant.filter_spec.sample_time_s),
        "lq_level_weight": float(plant.weights.q),
        "lq_reservoir_weight": float(plant.weights.r_reservoir),
        "lq_rate_weight": float(plant.weights.rho),
        "kalman_r1": float(plant.kalman_r1),
        "kalman_r2": float(plant.kalman_r2),
        "kalman_reading": plant.kalman_reading,
    }


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

    ``levels``, ``commands`` and ``applied`` each have one column per
    ``(node, block)`` pair, ordered node-major: column
    ``(n - 1) * blocks + j``. Their rows run over ``(step, node)`` in the
    same order :func:`faircanal.closedloop.simulate_closed_loop` returns
    them, flattened.

    The level arrays are one step longer than the flow arrays, because a
    run of ``K`` steps produces ``K + 1`` levels: the one it starts from
    and one after each step. All of them are carried. Dropping the last
    left the level at the end of the horizon unconstrained by C7, which is
    a silent hole rather than a saving - the row costs nothing and the
    level it bounds is a real one.

    ``applied`` is the flow that reaches the gate, after the filter;
    ``commands`` is what the controller asked for. See the module
    docstring for which belongs in which constraint.

    ``delivery`` is the same ``Gamma`` every gate uses - filter after
    hold - so a user's delivered flow is ``delivery @ v_i`` whatever gate
    they sit on.
    """

    levels: np.ndarray
    commands: np.ndarray
    applied: np.ndarray
    baseline_levels: np.ndarray
    baseline_commands: np.ndarray
    baseline_applied: np.ndarray
    delivery: np.ndarray
    steps: int
    blocks: int
    steps_per_block: int
    size: int
    transport_lag: tuple[int, ...]
    storage_area: tuple[float, ...]
    filter_order: int
    filter_cutoff_rad_per_s: float
    control_signature: dict

    def __post_init__(self) -> None:
        columns = self.size * self.blocks
        flow_rows = self.steps * self.size
        level_rows = (self.steps + 1) * self.size
        for name, matrix, shape in (
            ("levels", self.levels, (level_rows, columns)),
            ("commands", self.commands, (flow_rows, columns)),
            ("applied", self.applied, (flow_rows, columns)),
        ):
            if matrix.shape != shape:
                raise PlantError(f"{name} has shape {matrix.shape}, expected {shape}")
        for name, vector, length in (
            ("baseline_levels", self.baseline_levels, level_rows),
            ("baseline_commands", self.baseline_commands, flow_rows),
            ("baseline_applied", self.baseline_applied, flow_rows),
        ):
            if vector.shape != (length,):
                raise PlantError(
                    f"{name} has shape {vector.shape}, expected {(length,)}"
                )
        if self.delivery.shape != (self.steps, self.blocks):
            raise PlantError(
                f"delivery has shape {self.delivery.shape}, expected "
                f"{(self.steps, self.blocks)}"
            )
        for name, values in (
            ("transport_lag", self.transport_lag),
            ("storage_area", self.storage_area),
        ):
            if len(values) != self.size:
                raise PlantError(
                    f"{name} has {len(values)} entries for {self.size} reaches"
                )

    @property
    def level_steps(self) -> int:
        """How many level samples a run produces: one more than its steps."""
        return self.steps + 1

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
        """Level deviations for a whole table of orders, ``(steps + 1, size)``."""
        value = self.baseline_levels + self.levels @ self.flatten_orders(orders)
        return value.reshape(self.level_steps, self.size)

    def commands_at(self, orders: np.ndarray) -> np.ndarray:
        """Gate flows the controller asked for, ``(steps, size)``."""
        value = self.baseline_commands + self.commands @ self.flatten_orders(orders)
        return value.reshape(self.steps, self.size)

    def applied_at(self, orders: np.ndarray) -> np.ndarray:
        """Gate flows that actually reach the canal, ``(steps, size)``.

        This is the ``u_n[k]`` the conveyance limit C5 and the travel rate
        C6 are written on: the flow through the gate, after the filter.
        """
        value = self.baseline_applied + self.applied @ self.flatten_orders(orders)
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

    def peak_applied_gain(self) -> np.ndarray:
        """The same, for the flow that actually reaches the gate.

        Reported next to :meth:`peak_command_gain` because the difference
        between them is the filter, and because C5 is written on this one.
        """
        shaped = self.applied.reshape(
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
    baseline_levels = baseline.levels.reshape(-1).copy()
    baseline_commands = baseline.commanded.reshape(-1).copy()
    baseline_applied = baseline.applied.reshape(-1).copy()

    hold = hold_matrix(steps, blocks, steps_per_block)
    columns_levels = np.zeros(((steps + 1) * size, size * blocks))
    columns_commands = np.zeros((steps * size, size * blocks))
    columns_applied = np.zeros((steps * size, size * blocks))

    for node in range(1, size + 1):
        for block in range(blocks):
            offtake = np.zeros((steps, size))
            offtake[:, node - 1] = hold[:, block]
            run = simulate_closed_loop(loop, initial_levels, offtake, law=law)
            index = (node - 1) * blocks + block
            columns_levels[:, index] = run.levels.reshape(-1) - baseline_levels
            columns_commands[:, index] = (
                run.commanded.reshape(-1) - baseline_commands
            )
            columns_applied[:, index] = run.applied.reshape(-1) - baseline_applied

    return ResponseMap(
        levels=columns_levels,
        commands=columns_commands,
        applied=columns_applied,
        baseline_levels=baseline_levels,
        baseline_commands=baseline_commands,
        baseline_applied=baseline_applied,
        delivery=delivery_operator(
            plant.filter_spec, steps, blocks, steps_per_block
        ),
        steps=steps,
        blocks=blocks,
        steps_per_block=steps_per_block,
        size=size,
        transport_lag=tuple(int(pool.tau) for pool in plant.plant_models),
        storage_area=storage_areas(plant),
        filter_order=int(plant.filter_spec.order),
        filter_cutoff_rad_per_s=float(plant.filter_spec.cutoff_rad_per_s),
        control_signature=control_signature(plant),
    )
