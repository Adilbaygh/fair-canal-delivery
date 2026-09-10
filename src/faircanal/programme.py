"""The whole model as one linear programme with a fraction per user.

This is where the canal, the filter, the controller, the orders and the
fairness measure meet. Everything before it produced pieces; this module
writes down the polytope of C1 to C11 and the affine fraction ``r_i`` that
the lexicographic procedure of :mod:`faircanal.leximin` maximises.

The decision vector
-------------------
``z[i * blocks + j]`` is user ``i``'s order for block ``j``, as a
**deviation** from what it draws nominally. The pool models were
identified at the published steady state, so that state is the linear
model's zero and a user who orders nothing is a user drawing its usual
amount. Everything the user experiences stays absolute - the demand, the
delivered volume, the ratio - and the nominal enters those as a constant.

The closed loop is substituted, not relaxed
-------------------------------------------
The model document writes C10 as an equality constraint. Substituting the
response map instead - so that the levels and the commands are affine
functions of ``z`` and never appear as variables - is the same feasible
set with far fewer variables, and it has one consequence worth stating:
because the pool models integrate their net flow at exactly the rate their
backwater area says, the level bound C7 already bounds the stored volume.
So the slow-layer storage state C9 and its reconciliation band C9' are
implied by C7 rather than added, as long as the conveyance loss is one.
``test_the_storage_state_is_implied_by_the_level_band`` checks that
implication rather than assuming it, and the loss stays open until its
formula can be read from its source.

What each constraint is doing
-----------------------------
========  ===================================================
C1        bounds on ``z``: a user cannot draw below nothing,
          nor above its own outlet
C2        the delivered flow itself cannot go negative - the
          filter's impulse response dips below zero, so this
          does not come for free
C3        the volume budget, with a declared delivery
          allowance
C4        nothing is reshaped before it is announced: the
          deviation is pinned at zero in earlier blocks
C5        every gate stays between shut and the reach's
          conveyance
C6        no gate moves faster than it can
C7        every level stays inside its band
C8        the users cannot draw more than the source releases,
          from the step the restriction takes effect
C11       the first steps may be exempted from C5 to C7, for
          the transient the controller's feed-forward makes
          when it learns the whole future at once
========  ===================================================

and the cap ``r_i <= 1``, which is not decoration: without it, pouring
extra water on an already-satisfied user raises no minimum but does show
as improvement at later stages, and over-delivery becomes a way to score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from faircanal.leximin import RatioProgramme, TieBreak
from faircanal.plant import ResponseMap
from faircanal.scenario import Scenario, ScenarioError

__all__ = [
    "ProgrammeError",
    "Programme",
    "assemble",
    "movement_tie_break",
]


class ProgrammeError(ValueError):
    """Raised when a scenario and a response map cannot be put together."""


@dataclass(frozen=True)
class Programme:
    """The linear programme, and the means to read a solution of it back.

    ``ratio`` is what :mod:`faircanal.leximin` consumes. Everything else is
    here so that a decision vector can be turned back into flows, levels
    and volumes without rebuilding anything.
    """

    scenario: Scenario
    response: ResponseMap
    ratio: RatioProgramme
    spread: sparse.csr_matrix
    level_rows: np.ndarray
    command_rows: np.ndarray
    row_counts: dict

    @property
    def steps(self) -> int:
        return self.response.steps

    @property
    def blocks(self) -> int:
        return self.response.blocks

    @property
    def size(self) -> int:
        return self.response.size

    def orders_at(self, z: np.ndarray) -> np.ndarray:
        """The decision as a ``(user, block)`` table of deviations."""
        return np.asarray(z, dtype=float).reshape(self.scenario.n_users, self.blocks)

    def absolute_orders_at(self, z: np.ndarray) -> np.ndarray:
        """The same table, as the flow each user actually asks to draw."""
        nominal = np.array(
            [user.nominal_m3_s for user in self.scenario.users]
        ).reshape(-1, 1)
        return self.orders_at(z) + nominal

    def delivered_at(self, z: np.ndarray) -> np.ndarray:
        """Absolute delivered flow, ``(steps, user)``.

        The nominal draw is already flowing when the run starts, so it
        arrives undelayed; only the deviation goes through the filter.
        """
        nominal = np.array([user.nominal_m3_s for user in self.scenario.users])
        return nominal + self.response.delivery @ self.orders_at(z).T

    def volumes_at(self, z: np.ndarray) -> np.ndarray:
        """Volume delivered inside each user's own window, m^3."""
        delivered = self.delivered_at(z)
        return np.array(
            [
                delivered[user.window[0] : user.window[1] + 1, index].sum()
                * self.scenario.dt_s
                for index, user in enumerate(self.scenario.users)
            ]
        )

    def levels_at(self, z: np.ndarray) -> np.ndarray:
        value = self.response.baseline_levels + self.level_rows @ np.asarray(z)
        return value.reshape(self.steps, self.size)

    def commands_at(self, z: np.ndarray) -> np.ndarray:
        value = self.response.baseline_commands + self.command_rows @ np.asarray(z)
        return value.reshape(self.steps, self.size)

    def ratios_at(self, z: np.ndarray) -> np.ndarray:
        return self.ratio.ratios_at(np.asarray(z, dtype=float))


def _spread_matrix(scenario: Scenario, size: int, blocks: int) -> sparse.csr_matrix:
    """Add up the users at each gate, block by block."""
    rows, cols = [], []
    for index, user in enumerate(scenario.users):
        for block in range(blocks):
            rows.append((user.node - 1) * blocks + block)
            cols.append(index * blocks + block)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(size * blocks, len(scenario.users) * blocks)
    )


def _path_loss(scenario: Scenario, node: int) -> float:
    """How much has to be released for one unit to arrive at *node*."""
    if scenario.loss_factor is None:
        return 1.0
    factor = 1.0
    for step in scenario.network.path_to_root(node):
        efficiency = scenario.loss_factor[step - 1]
        if not 0.0 < efficiency <= 1.0:
            raise ProgrammeError(
                f"reach {step}: a conveyance efficiency of {efficiency} is not a "
                f"fraction"
            )
        factor /= efficiency
    return factor


def assemble(scenario: Scenario, response: ResponseMap) -> Programme:
    """Write the whole model down as a :class:`RatioProgramme`."""
    if response.size != scenario.network.size:
        raise ProgrammeError(
            f"the map covers {response.size} reaches and the scenario "
            f"{scenario.network.size}"
        )
    if response.blocks != scenario.blocks:
        raise ProgrammeError(
            f"the map covers {response.blocks} blocks and the scenario "
            f"{scenario.blocks}"
        )
    if response.steps != scenario.horizon:
        raise ProgrammeError(
            f"the map runs {response.steps} steps and the scenario "
            f"{scenario.horizon}"
        )

    users = scenario.users
    n_users, blocks, size = len(users), scenario.blocks, response.size
    steps, dt = response.steps, scenario.dt_s
    block_seconds = dt * scenario.steps_per_block
    n_vars = n_users * blocks

    spread = _spread_matrix(scenario, size, blocks)
    level_rows = response.levels @ spread
    command_rows = response.commands @ spread
    gamma = response.delivery

    nominal = np.array([user.nominal_m3_s for user in users])

    # --- the fractions -----------------------------------------------------
    ratio_rows = np.zeros((n_users, n_vars))
    ratio_offset = np.zeros(n_users)
    for index, user in enumerate(users):
        first, last = user.window
        window = gamma[first : last + 1, :].sum(axis=0) * dt
        ratio_rows[index, index * blocks : (index + 1) * blocks] = window / user.demand_m3
        ratio_offset[index] = user.nominal_volume_m3(dt) / user.demand_m3

    # --- bounds: C1 and C4 -------------------------------------------------
    bounds: list[tuple[float | None, float | None]] = []
    for user in users:
        ceiling = (
            None
            if user.max_order_m3_s is None
            else user.max_order_m3_s - user.nominal_m3_s
        )
        for block in range(blocks):
            if block < user.announced_block:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((-user.nominal_m3_s, ceiling))

    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    counts: dict[str, int] = {}

    def add(block_rows: np.ndarray, block_values: np.ndarray, label: str) -> None:
        if block_rows.size == 0:
            counts[label] = 0
            return
        rows.append(np.atleast_2d(block_rows))
        values.append(np.atleast_1d(block_values))
        counts[label] = int(np.atleast_2d(block_rows).shape[0])

    # --- C2: the delivered flow cannot go negative -------------------------
    c2 = np.zeros((n_users * steps, n_vars))
    c2_b = np.zeros(n_users * steps)
    for index, user in enumerate(users):
        block = slice(index * blocks, (index + 1) * blocks)
        c2[index * steps : (index + 1) * steps, block] = -gamma
        c2_b[index * steps : (index + 1) * steps] = user.nominal_m3_s
    add(c2, c2_b, "C2 delivered flow non-negative")

    # --- C3: the volume budget --------------------------------------------
    c3 = np.zeros((n_users, n_vars))
    c3_b = np.zeros(n_users)
    for index, user in enumerate(users):
        c3[index, index * blocks : (index + 1) * blocks] = block_seconds
        c3_b[index] = (1.0 + user.overshoot) * user.demand_m3 - (
            user.nominal_m3_s * block_seconds * blocks
        )
    add(c3, c3_b, "C3 volume budget")

    # --- C5, C6, C7: what the canal may do --------------------------------
    warm = scenario.limits.warm_up_steps
    if warm >= steps:
        raise ProgrammeError(
            f"a warm-up of {warm} steps leaves nothing of a {steps}-step run"
        )
    active = np.arange(warm, steps)
    keep = (active[:, None] * size + np.arange(size)[None, :]).reshape(-1)

    command_bounds = scenario.limits.command_bounds
    low = np.array([bound[0] for bound in command_bounds])
    high = np.array([bound[1] for bound in command_bounds])
    base_u = response.baseline_commands
    a_u = command_rows[keep]
    b_u = base_u[keep]
    tiled_low = np.tile(low, active.size)
    tiled_high = np.tile(high, active.size)
    add(
        np.vstack([a_u, -a_u]),
        np.concatenate([tiled_high - b_u, b_u - tiled_low]),
        "C5 gate flow inside the reach's conveyance",
    )

    band_low = np.array([band[0] for band in scenario.limits.level_band_m])
    band_high = np.array([band[1] for band in scenario.limits.level_band_m])
    base_y = response.baseline_levels
    a_y = level_rows[keep]
    b_y = base_y[keep]
    add(
        np.vstack([a_y, -a_y]),
        np.concatenate(
            [np.tile(band_high, active.size) - b_y, b_y - np.tile(band_low, active.size)]
        ),
        "C7 level inside its band",
    )

    if active.size > 1:
        later = keep[size:]
        earlier = keep[:-size]
        step_rows = command_rows[later] - command_rows[earlier]
        step_base = base_u[later] - base_u[earlier]
        travel = np.tile(np.array(scenario.limits.travel_rate_m3_s), active.size - 1)
        add(
            np.vstack([step_rows, -step_rows]),
            np.concatenate([travel - step_base, travel + step_base]),
            "C6 gate travel rate",
        )
    else:
        counts["C6 gate travel rate"] = 0

    # --- C8: the source ----------------------------------------------------
    c8 = np.zeros((steps, n_vars))
    released = 0.0
    for index, user in enumerate(users):
        factor = _path_loss(scenario, user.node)
        c8[:, index * blocks : (index + 1) * blocks] = factor * gamma
        released += factor * user.nominal_m3_s
    add(
        c8,
        np.array(scenario.source_profile(), dtype=float) - released,
        "C8 source availability",
    )

    # --- the cap on the fraction ------------------------------------------
    add(
        ratio_rows.copy(),
        scenario.ratio_cap - ratio_offset,
        "ratio capped at one",
    )

    a_ub = sparse.csr_matrix(np.vstack(rows)) if rows else sparse.csr_matrix((0, n_vars))
    b_ub = np.concatenate(values) if values else np.zeros(0)

    programme = RatioProgramme(
        n_vars=n_vars,
        a_ub=a_ub,
        b_ub=b_ub,
        ratio_rows=sparse.csr_matrix(ratio_rows),
        ratio_offset=ratio_offset,
        weights=np.array([user.weight for user in users]),
        names=tuple(user.name for user in users),
        bounds=tuple(bounds),
    )
    return Programme(
        scenario=scenario,
        response=response,
        ratio=programme,
        spread=spread,
        level_rows=level_rows,
        command_rows=command_rows,
        row_counts=counts,
    )


def movement_tie_break(programme: Programme) -> TieBreak:
    """Pick the smoothest of the lexicographically optimal schedules.

    Section 7.4: among every decision that reaches the same fractions,
    take the one that moves the gates least, measured as the total
    absolute step-to-step change. Two things at once - the answer becomes
    reproducible, and the schedule that is reported is the one a canal
    would rather run.

    The absolute value is linearised the usual way, with one auxiliary
    variable per gate and step bounding the movement from both sides.
    """
    steps, size = programme.steps, programme.size
    n_vars = programme.ratio.n_vars
    later = np.arange(size, steps * size)
    earlier = np.arange(0, (steps - 1) * size)
    difference = programme.command_rows[later] - programme.command_rows[earlier]
    base = (
        programme.response.baseline_commands[later]
        - programme.response.baseline_commands[earlier]
    )
    n_extra = difference.shape[0]

    identity = sparse.identity(n_extra, format="csr")
    upper = sparse.hstack([sparse.csr_matrix(difference), -identity], format="csr")
    lower = sparse.hstack([sparse.csr_matrix(-difference), -identity], format="csr")
    return TieBreak(
        n_extra=n_extra,
        a_ub=sparse.vstack([upper, lower], format="csr"),
        b_ub=np.concatenate([-base, base]),
        cost=np.concatenate([np.zeros(n_vars), np.ones(n_extra)]),
        bounds_extra=tuple((0.0, None) for _ in range(n_extra)),
    )
