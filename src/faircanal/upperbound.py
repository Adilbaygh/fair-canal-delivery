"""M1: the same canal with the controller taken out, as a bound.

Every other alternative in :mod:`faircanal.baselines` competes. This one
does not. It answers a different question, and the paper has to be able to
answer it before it claims anything: *how much of what the users do not
get is the fairness criterion's fault, and how much is the canal's?*

M1 removes the one thing this study is not allowed to touch - the
controller - and lets an optimiser choose the gate commands directly,
under the same physical limits as everybody else. Whatever it reaches is
what the canal could deliver if the control layer were perfect. If the
proposed method is already close to it, there is nothing left to win by
redesigning the controller, and the paper can say so with a number. If it
is far from it, the paper has to say that too.

What is removed, and what is emphatically not
---------------------------------------------
Removed: the structured LQ feedback law and its observer. The gate
command becomes a free variable at every step.

Kept, unchanged: the pool models, the transport delays, the conveyance
and travel limits, the level band, the source restriction, the volume
budget, the deadline windows - and **the low-pass filter**. The filter is
not part of the controller; it sits between the controller and the plant
and in front of every planned offtake, and this study never touches it.
Keeping it is also what makes the bound a bound: a schedule the proposed
method can produce is feasible here with the controller's own commands
plugged in as the free variable, so this feasible set contains that one,
and its optimum cannot be worse.
``test_the_proposed_schedule_is_feasible_here`` checks that containment
on the real canal rather than leaving it as an argument.

How it is written down
----------------------
The closed loop was substituted away in :mod:`faircanal.programme`,
because there the levels and commands were affine in the order and could
be folded into the rows. Here they cannot: the command is free, so the
plant has to appear as itself. It does, as two recursions written as
equality constraints, both sparse:

* the filter, as its own difference equation, turning the commanded flow
  into the applied one - seven coefficients a row, not a full triangular
  matrix;
* the pool, as the third-order difference equation of the source, in the
  applied flows and the levels.

The decision vector is ``[z, u, a, y]``: the orders, the commanded gate
flows, the applied gate flows, and the levels. Everything is a deviation
from the published steady state, as everywhere else in this package, so
the run starts from rest and stays there when nothing is ordered.

Why the conveyance and travel limits act on the commanded flow
--------------------------------------------------------------
Because that is where they act in the programme this is a bound for. The
comparison is only worth making if the two feasible sets differ in one
thing, and the one thing here is the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from faircanal.leximin import RatioProgramme
from faircanal.plant import CanalPlant
from faircanal.programme import Programme

__all__ = [
    "UpperBoundError",
    "FreeGatePlant",
    "free_gate_programme",
]


class UpperBoundError(ValueError):
    """Raised when the free-gate programme cannot be built."""


@dataclass(frozen=True)
class FreeGatePlant:
    """The free-gate programme, and where each quantity sits inside it.

    ``ratio`` is the programme itself, ready for the lexicographic
    procedure. The rest is bookkeeping: the decision vector is one long
    array and these are the offsets that turn a solution back into
    schedules, commands and levels.
    """

    ratio: RatioProgramme
    steps: int
    size: int
    blocks: int
    n_users: int
    offset_command: int
    offset_applied: int
    offset_level: int

    @property
    def n_orders(self) -> int:
        return self.n_users * self.blocks

    def orders_of(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[: self.n_orders]

    def _grid(self, x: np.ndarray, offset: int) -> np.ndarray:
        block = np.asarray(x, dtype=float)[offset : offset + self.steps * self.size]
        return block.reshape(self.steps, self.size)

    def commands_of(self, x: np.ndarray) -> np.ndarray:
        return self._grid(x, self.offset_command)

    def applied_of(self, x: np.ndarray) -> np.ndarray:
        return self._grid(x, self.offset_applied)

    def levels_of(self, x: np.ndarray) -> np.ndarray:
        return self._grid(x, self.offset_level)


def _index(offset: int, step: int, node: int, size: int) -> int:
    """Column of a per-step, per-reach quantity, step-major as elsewhere."""
    return offset + step * size + (node - 1)


def _filter_rows(
    spec, steps: int, size: int, offset_command: int, offset_applied: int, columns: int
):
    """The filter's own difference equation, one row per reach and step.

    ``den[0] a[k] + sum_{j>0} den[j] a[k-j] = sum_i num[i] u[k-i]``, with
    everything before the start at zero because the filter is started from
    rest - the same state the closed-loop simulation starts it in.
    """
    numerator, denominator = spec.coefficients()
    numerator = np.asarray(numerator, dtype=float).ravel()
    denominator = np.asarray(denominator, dtype=float).ravel()
    if denominator[0] == 0.0:
        raise UpperBoundError("the filter's leading denominator coefficient is zero")

    rows, cols, values = [], [], []
    row = 0
    for step in range(steps):
        for node in range(1, size + 1):
            for lag, coefficient in enumerate(denominator):
                if step - lag < 0 or coefficient == 0.0:
                    continue
                rows.append(row)
                cols.append(_index(offset_applied, step - lag, node, size))
                values.append(float(coefficient))
            for lag, coefficient in enumerate(numerator):
                if step - lag < 0 or coefficient == 0.0:
                    continue
                rows.append(row)
                cols.append(_index(offset_command, step - lag, node, size))
                values.append(-float(coefficient))
            row += 1
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(steps * size, columns)
    ), np.zeros(steps * size)


def _pool_rows(
    programme: Programme,
    plant: CanalPlant,
    offtake_rows: sparse.csr_matrix,
    offset_applied: int,
    offset_level: int,
    columns: int,
):
    """The pool recursion of the source, as equality constraints.

    For the third-order model, collecting the homogeneous part,

        Y[k+1] = (1 + a1 + a2) Y[k] - (2 a1 + a2) Y[k-1] + a1 Y[k-2]
                 + b1 A[k-tau] - b2 A[k-tau-1] + b3 A[k-tau-2]
                 - c1 W[k] + c2 W[k-1] - c3 W[k-2],

    where ``A`` is the applied inflow of that reach and ``W`` is
    everything leaving at its downstream end: the applied inflows of its
    children plus the offtake the users at its gate draw.

    Which level is which
    --------------------
    ``Y[k]`` is the level at the **start** of step ``k``, because that is
    the one the substituted programme carries and bounds. Getting this
    wrong is not a small error and it is not a loud one: the two readings
    differ by a single step of level change, about a centimetre here, so
    the bound would quietly be applied to the wrong quantity and every
    number would still look plausible.
    ``test_the_free_gate_plant_reproduces_the_closed_loop`` pins the two
    together on the real canal.

    So the first row of each reach is not a recursion at all but the
    initial condition, and the recursion fills in the rest. The level
    before the run is held at that same initial value, exactly as the
    simulation holds it, and where such a term appears it moves to the
    right-hand side as the constant it is.

    The offtake is affine in the order, so those coefficients land in the
    order columns; that is the only place the two halves of the decision
    vector meet.
    """
    network = programme.scenario.network
    steps, size = programme.steps, programme.response.size
    indptr = offtake_rows.indptr
    indices = offtake_rows.indices
    data = offtake_rows.data
    initial = np.asarray(programme.response.baseline_levels, dtype=float).reshape(
        steps, size
    )[0]

    rows, cols, values = [], [], []
    constants = np.zeros(steps * size)
    row = 0

    for reach, model in zip(network.reaches, plant.plant_models):
        node = reach.node
        children = network.children(node)
        start = float(initial[node - 1])

        def add(step: int, offset: int, target: int, coefficient: float) -> None:
            if step < 0 or coefficient == 0.0:
                return
            rows.append(row)
            cols.append(_index(offset, step, target, size))
            values.append(float(coefficient))

        def add_level(step: int, coefficient: float) -> None:
            """A level term, or the constant it becomes before the start."""
            if coefficient == 0.0:
                return
            if step < 0:
                constants[row] += coefficient * start
                return
            rows.append(row)
            cols.append(_index(offset_level, step, node, size))
            values.append(float(coefficient))

        # The initial condition, as its own row.
        add_level(0, 1.0)
        constants[row] -= start
        row += 1

        for step in range(steps - 1):
            add_level(step + 1, 1.0)
            drawn: list[tuple[int, float]] = []

            if model.order == 1:
                b1 = model.b[0]
                c1 = model.c[0]
                add_level(step, -1.0)
                add(step - model.tau - model.tau_bar, offset_applied, node, -b1)
                drawn.append((step - model.tau_bar, c1))
            else:
                b1, b2, b3 = model.b
                c1, c2, c3 = model.c
                a1, a2 = model.alpha
                add_level(step, -(1.0 + a1 + a2))
                add_level(step - 1, 2.0 * a1 + a2)
                add_level(step - 2, -a1)
                add(step - model.tau, offset_applied, node, -b1)
                add(step - model.tau - 1, offset_applied, node, b2)
                add(step - model.tau - 2, offset_applied, node, -b3)
                drawn.extend([(step, c1), (step - 1, -c2), (step - 2, c3)])

            # Everything leaving at the downstream end, with its sign: the
            # children's applied inflows, and the offtake, which is affine
            # in the order and so lands in the order columns.
            for when, coefficient in drawn:
                if when < 0 or coefficient == 0.0:
                    continue
                for child in children:
                    add(when, offset_applied, child, coefficient)
                source = when * size + node - 1
                for entry in range(indptr[source], indptr[source + 1]):
                    rows.append(row)
                    cols.append(int(indices[entry]))
                    values.append(coefficient * float(data[entry]))
            row += 1

    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(row, columns)
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, -constants[:row]


def _offtake_rows(programme: Programme) -> sparse.csr_matrix:
    """Offtake drawn at each reach and step, as a function of the order.

    The delivery operator already carries the zero-order hold and the
    filter, so this is only a matter of sending each user's blocks to the
    gate it draws from and stacking the reaches step by step.
    """
    steps, size = programme.steps, programme.response.size
    blocks = programme.blocks
    gamma = np.asarray(programme.response.delivery, dtype=float)

    rows, cols, values = [], [], []
    for step in range(steps):
        for node in range(1, size + 1):
            for block in range(blocks):
                coefficient = gamma[step, block]
                if coefficient == 0.0:
                    continue
                rows.append(step * size + node - 1)
                cols.append((node - 1) * blocks + block)
                values.append(coefficient)
    spread_to_steps = sparse.csr_matrix(
        (values, (rows, cols)), shape=(steps * size, size * blocks)
    )
    return sparse.csr_matrix(spread_to_steps @ programme.spread)


def free_gate_programme(programme: Programme, plant: CanalPlant) -> FreeGatePlant:
    """Write the same experiment down with the gate commands set free.

    The rows that only involve the order - the delivered flow staying
    non-negative, the volume budget, the source restriction and the cap on
    the fraction - are taken from the substituted programme unchanged, so
    the two differ in the controller and in nothing else. The rows that
    involved the controller are replaced: the conveyance limit and the
    level band become bounds on their own variables, the travel rate
    becomes a row on consecutive commands, and the plant becomes the two
    recursions above.
    """
    scenario = programme.scenario
    if scenario.network.size != programme.response.size:
        raise UpperBoundError("the scenario and the response map disagree on the size")
    if plant.size != scenario.network.size:
        raise UpperBoundError("the plant and the scenario disagree on the size")

    steps, size = programme.steps, programme.response.size
    blocks, n_users = programme.blocks, scenario.n_users
    n_orders = n_users * blocks
    grid = steps * size

    offset_command = n_orders
    offset_applied = offset_command + grid
    offset_level = offset_applied + grid
    columns = offset_level + grid

    # --- the rows that do not mention the controller ----------------------
    keep = {
        "C2 delivered flow non-negative",
        "C3 volume budget",
        "C8 source availability",
        "ratio capped at one",
    }
    first = 0
    order_only, order_values, counts = [], [], {}
    for name, rows in programme.row_counts.items():
        last = first + rows
        if name in keep and rows:
            order_only.append(programme.ratio.a_ub[first:last])
            order_values.append(programme.ratio.b_ub[first:last])
            counts[name] = rows
        first = last
    if first != programme.ratio.a_ub.shape[0]:
        raise UpperBoundError("the row counts do not add up to the programme's rows")

    blocks_ub = [
        sparse.hstack([piece, sparse.csr_matrix((piece.shape[0], columns - n_orders))])
        for piece in order_only
    ]
    values_ub = list(order_values)

    # --- C6: no gate moves faster than it can -----------------------------
    warm = scenario.limits.warm_up_steps
    if warm >= steps:
        raise UpperBoundError(
            f"a warm-up of {warm} steps leaves nothing of a {steps}-step run"
        )
    travel = np.asarray(scenario.limits.travel_rate_m3_s, dtype=float)
    rows, cols, values = [], [], []
    limits = []
    row = 0
    for step in range(warm + 1, steps):
        for node in range(1, size + 1):
            for sign in (1.0, -1.0):
                rows.extend([row, row])
                cols.extend(
                    [
                        _index(offset_command, step, node, size),
                        _index(offset_command, step - 1, node, size),
                    ]
                )
                values.extend([sign, -sign])
                limits.append(travel[node - 1])
                row += 1
    if row:
        blocks_ub.append(sparse.csr_matrix((values, (rows, cols)), shape=(row, columns)))
        values_ub.append(np.array(limits))
    counts["C6 gate travel rate"] = row

    a_ub = sparse.vstack(blocks_ub, format="csr")
    b_ub = np.concatenate(values_ub)

    # --- the plant, as equalities -----------------------------------------
    offtake = _offtake_rows(programme)
    filter_a, filter_b = _filter_rows(
        plant.filter_spec, steps, size, offset_command, offset_applied, columns
    )
    pool_a, pool_b = _pool_rows(
        programme, plant, offtake, offset_applied, offset_level, columns
    )
    a_eq = sparse.vstack([filter_a, pool_a], format="csr")
    b_eq = np.concatenate([filter_b, pool_b])

    # --- bounds: C1 and C4 on the order, C5 and C7 on their own variables --
    command_bounds = scenario.limits.command_bounds
    band = scenario.limits.level_band_m
    bounds: list[tuple[float | None, float | None]] = list(programme.ratio.bounds)
    for step in range(steps):
        for node in range(1, size + 1):
            bounds.append(
                (None, None) if step < warm else command_bounds[node - 1]
            )
    bounds.extend([(None, None)] * grid)  # the applied flow is determined
    for step in range(steps):
        for node in range(1, size + 1):
            bounds.append((None, None) if step < warm else band[node - 1])

    ratio_rows = sparse.hstack(
        [programme.ratio.ratio_rows, sparse.csr_matrix((n_users, columns - n_orders))],
        format="csr",
    )
    free = RatioProgramme(
        n_vars=columns,
        a_ub=a_ub,
        b_ub=b_ub,
        ratio_rows=ratio_rows,
        ratio_offset=programme.ratio.ratio_offset,
        weights=programme.ratio.weights,
        names=programme.ratio.names,
        bounds=tuple(bounds),
        a_eq=a_eq,
        b_eq=b_eq,
    )
    return FreeGatePlant(
        ratio=free,
        steps=steps,
        size=size,
        blocks=blocks,
        n_users=n_users,
        offset_command=offset_command,
        offset_applied=offset_applied,
        offset_level=offset_level,
    )
