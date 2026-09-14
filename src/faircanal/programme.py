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
response map instead - so that the levels, the commands and the applied
flows are affine functions of ``z`` and never appear as variables - is the
*same feasible set* with far fewer variables.

That is a claim, not a convenience, so it is checked rather than asserted:
``test_c10_substitution_equals_equality_rows`` builds the equality form on
a small instance and requires the two polytopes to accept and refuse the
same points.

The storage state is carried, not assumed away
----------------------------------------------
An earlier version left C9 and C9' out on the argument that the level
bound C7 already bounds the stored volume. It does not, and the argument
had a second flaw: the test it cited had never been written. Two accounts
of the same water exist here - a slow one in volume and a fast one in
level - and they are not the same account. An IDZ pool is not a level
pool: its storage is not the downstream level times an area, because the
surface tilts when the flow changes and the wedge that tilting holds is
exactly what the model's delay and its zero represent.

So both are written down, and C9' ties them together with a band whose
width was measured rather than chosen - see ``Limits.band_tolerance_m``,
whose default is the measured ``BAND_TOLERANCE_M``, and the measurement
itself in ``scripts/check_storage_band.py``. The area is the backwater area the
pool actually integrates at, not the water surface; on this canal the two
differ by up to three quarters.

Because every block volume is itself affine in ``z``, the running storage
total is affine too, so C9 and C9' cost rows and no new variables. The
state stays a state: the row for block ``j`` carries every block before
it, so a slack of storage cannot reappear afresh each block and make water
out of nothing.

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
          deviation is pinned at zero in earlier blocks, so
          the announced nominal order stands
C5        every gate passes between nothing and the smaller
          of what the reach conveys and what the gate itself
          lets through - written on the **applied** flow,
          which is the water, not on the command
C5'       and the command stays inside the range the pool
          models were identified over, which is a different
          constraint on a different signal: the command
          swings several times harder than the flow
C6        no gate moves faster than it can, again on the
          applied flow
C7        every level stays inside its band, at every one of
          the K+1 samples a run of K steps produces
C8        the users cannot draw more than the source releases,
          from the step the restriction takes effect
C9        pool storage is a state with a floor and a ceiling,
          accumulated block by block from the volumes that
          entered and left
C9'       and that state agrees with the level account to
          within a declared, measured band
C11       the first steps may be exempted from C7 - and from
          C7 only - for the transient the controller's
          feed-forward makes when it learns the whole future
          at once
========  ===================================================

and the cap ``r_i <= 1``, which is not decoration: without it, pouring
extra water on an already-satisfied user raises no minimum but does show
as improvement at later stages, and over-delivery becomes a way to score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from faircanal.config import CONVEYANCE_EFFICIENCY
from faircanal.leximin import RatioProgramme, TieBreak
from faircanal.plant import ResponseMap
from faircanal.scenario import Scenario

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
    applied_rows: np.ndarray
    storage_rows: np.ndarray
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
        """Level deviations, ``(steps + 1, size)``."""
        value = self.response.baseline_levels + self.level_rows @ np.asarray(z)
        return value.reshape(self.response.level_steps, self.size)

    def commands_at(self, z: np.ndarray) -> np.ndarray:
        """What the controller asks each gate for, ``(steps, size)``."""
        value = self.response.baseline_commands + self.command_rows @ np.asarray(z)
        return value.reshape(self.steps, self.size)

    def applied_at(self, z: np.ndarray) -> np.ndarray:
        """What each gate actually passes, ``(steps, size)``."""
        value = self.response.baseline_applied + self.applied_rows @ np.asarray(z)
        return value.reshape(self.steps, self.size)

    def storage_at(self, z: np.ndarray) -> np.ndarray:
        """Pool storage relative to its nominal, ``(size, blocks + 1)`` m^3.

        Block ``0`` is the state the run starts from, so it is zero by
        construction; block ``j`` carries everything that entered and left
        before it.
        """
        value = self.storage_rows @ np.asarray(z)
        return value.reshape(self.size, self.blocks + 1)

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


def _block_mean_levels(response: ResponseMap, level_rows: np.ndarray) -> np.ndarray:
    """Each pool's mean level over each block, as rows in ``z``.

    One row per ``(pool, block)`` in pool-major order, so it lines up with
    the storage rows C9' compares it against. The block index runs to
    ``blocks`` inclusive because the storage state does too: block zero is
    where the run starts.
    """
    size, blocks = response.size, response.blocks
    per_block, level_steps = response.steps_per_block, response.level_steps
    out = np.zeros((size * (blocks + 1), level_rows.shape[1]))
    index = np.arange(level_steps) * size
    for pool in range(size):
        rows = level_rows[index + pool]
        for block in range(blocks + 1):
            first = min(block * per_block, level_steps - per_block)
            out[pool * (blocks + 1) + block] = rows[first : first + per_block].mean(
                axis=0
            )
    return out


def _storage_rows(
    scenario: Scenario,
    response: ResponseMap,
    applied_rows: np.ndarray,
    gamma: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Pool storage, relative to its nominal, as rows in ``z``.

    The recursion of C9 written out. Every block volume is affine in the
    decision, so the running total is affine too and the state needs rows
    rather than variables:

        S_p[j] - S_p[0] = sum over the blocks before j of
            eta_p V^{e,tau}_p - sum over children V^e - sum over users V^d

    ``V^e_n`` is the volume that passed gate ``n`` in a block, taken from
    the **applied** flow because that is the water; ``V^{e,tau}`` is the
    same shifted by the reach's transport lag, so water is credited to the
    pool when it arrives and not when it set off. A pool's outflow is the
    flow through the gate below it, and the most downstream pool has none:
    everything that reaches it leaves through its own offtake.

    The result is zero at block zero and at ``z = 0``, because the canal
    starts in the identified steady state where every deviation is zero.
    """
    size, blocks = response.size, response.blocks
    steps, per_block = response.steps, response.steps_per_block
    lags = response.transport_lag
    n_vars = applied_rows.shape[1]
    index = np.arange(steps) * size

    inflow = np.zeros((size, blocks, n_vars))
    delayed = np.zeros((size, blocks, n_vars))
    for pool in range(size):
        rows = applied_rows[index + pool]
        lag = lags[pool]
        for block in range(blocks):
            window = np.arange(block * per_block, (block + 1) * per_block)
            inflow[pool, block] = dt * rows[window].sum(axis=0)
            shifted = window - lag
            arrived = shifted[shifted >= 0]
            if arrived.size:
                delayed[pool, block] = dt * rows[arrived].sum(axis=0)

    drawn = np.zeros((size, blocks, n_vars))
    block_volume = np.array(
        [
            dt * gamma[block * per_block : (block + 1) * per_block].sum(axis=0)
            for block in range(blocks)
        ]
    )
    for order, user in enumerate(scenario.users):
        columns = slice(order * blocks, (order + 1) * blocks)
        for block in range(blocks):
            drawn[user.node - 1, block, columns] += block_volume[block]

    net = CONVEYANCE_EFFICIENCY * delayed - drawn
    net[1:] -= inflow[:-1]

    storage = np.zeros((size, blocks + 1, n_vars))
    for block in range(blocks):
        storage[:, block + 1] = storage[:, block] + net[:, block]
    return storage.reshape(size * (blocks + 1), n_vars)


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
    applied_rows = response.applied @ spread
    gamma = response.delivery

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

    # --- C5, C5', C6: what the canal may do -------------------------------
    #
    # The warm-up does not reach any of these. It exempts C7 and nothing
    # else: the level band is what an order-induced transient can
    # legitimately break while the feed-forward settles, and a gate that
    # cannot pass the water cannot pass it in the first fifteen minutes
    # either.
    warm = scenario.limits.warm_up_steps
    if warm >= steps:
        raise ProgrammeError(
            f"a warm-up of {warm} steps leaves nothing of a {steps}-step run"
        )

    def two_sided(rows, base, low, high, label, span):
        """Add ``low <= base + rows z <= high`` as two blocks of rows."""
        add(
            np.vstack([rows, -rows]),
            np.concatenate([np.tile(high, span) - base, base - np.tile(low, span)]),
            label,
        )

    flow_low = np.array([bound[0] for bound in scenario.limits.flow_bounds])
    flow_high = np.array([bound[1] for bound in scenario.limits.flow_bounds])
    two_sided(
        applied_rows,
        response.baseline_applied,
        flow_low,
        flow_high,
        "C5 gate flow inside the reach's conveyance",
        steps,
    )

    command_low = np.array([bound[0] for bound in scenario.limits.command_bounds])
    command_high = np.array([bound[1] for bound in scenario.limits.command_bounds])
    two_sided(
        command_rows,
        response.baseline_commands,
        command_low,
        command_high,
        "C5' command inside the linear model's range",
        steps,
    )

    if steps > 1:
        later = np.arange(size, steps * size)
        earlier = np.arange(0, (steps - 1) * size)
        step_rows = applied_rows[later] - applied_rows[earlier]
        step_base = (
            response.baseline_applied[later] - response.baseline_applied[earlier]
        )
        travel = np.tile(np.array(scenario.limits.travel_rate_m3_s), steps - 1)
        add(
            np.vstack([step_rows, -step_rows]),
            np.concatenate([travel - step_base, travel + step_base]),
            "C6 gate travel rate",
        )
    else:
        counts["C6 gate travel rate"] = 0

    # --- C7: the level band, and the one thing the warm-up frees ----------
    #
    # A run of K steps produces K+1 levels and all of them are bounded; the
    # last used to fall off the end, which left the level at the horizon
    # free for no reason anybody had written down.
    level_steps = response.level_steps
    active = np.arange(warm, level_steps)
    keep = (active[:, None] * size + np.arange(size)[None, :]).reshape(-1)
    band_low = np.array([band[0] for band in scenario.limits.level_band_m])
    band_high = np.array([band[1] for band in scenario.limits.level_band_m])
    two_sided(
        level_rows[keep],
        response.baseline_levels[keep],
        band_low,
        band_high,
        "C7 level inside its band",
        active.size,
    )

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

    # --- C9: storage is a state, with a floor and a ceiling ---------------
    storage_rows = _storage_rows(scenario, response, applied_rows, gamma, dt)
    pools = scenario.network.reaches
    area = np.array(response.storage_area)
    nominal_storage = area * np.array([reach.pool.target_level_m for reach in pools])
    full_storage = area * np.array([reach.pool.canal_depth_m for reach in pools])
    span = blocks + 1
    add(
        np.vstack([storage_rows, -storage_rows]),
        np.concatenate(
            [
                np.repeat(full_storage - nominal_storage, span),
                np.repeat(nominal_storage, span),
            ]
        ),
        "C9 pool storage between empty and full",
    )

    # --- C9': the two accounts of the same water agree to within a band ---
    mean_level = _block_mean_levels(response, level_rows)
    reconcile = storage_rows - np.repeat(area, span)[:, None] * mean_level
    epsilon = np.repeat(area * scenario.limits.band_tolerance_m, span)
    add(
        np.vstack([reconcile, -reconcile]),
        np.concatenate([epsilon, epsilon]),
        "C9' storage agrees with the level",
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
        applied_rows=applied_rows,
        storage_rows=storage_rows,
        row_counts=counts,
    )


def movement_tie_break(programme: Programme) -> TieBreak:
    """Pick the smoothest of the lexicographically optimal schedules.

    Section 7.4: among every decision that reaches the same fractions,
    take the one that moves the gates least, measured as the total
    absolute step-to-step change. Two things at once - the answer becomes
    reproducible, and the schedule that is reported is the one a canal
    would rather run.

    Measured on the **applied** flow, which is the water the gate passes.
    Minimising the command instead would smooth a signal nobody sees and
    leave the claim that the chosen schedule is the hydraulically smoothest
    one untrue.

    The absolute value is linearised the usual way, with one auxiliary
    variable per gate and step bounding the movement from both sides.
    """
    steps, size = programme.steps, programme.size
    n_vars = programme.ratio.n_vars
    later = np.arange(size, steps * size)
    earlier = np.arange(0, (steps - 1) * size)
    difference = programme.applied_rows[later] - programme.applied_rows[earlier]
    base = (
        programme.response.baseline_applied[later]
        - programme.response.baseline_applied[earlier]
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
