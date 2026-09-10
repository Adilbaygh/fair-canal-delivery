"""The comparisons this study's claim rests on, all on one polytope.

A fairness result is only worth as much as what it is compared against, so
every alternative here is solved over the *same* feasible set that
:mod:`faircanal.programme` assembles from C1 to C11. Nothing but the
objective changes between them. If the constraints differed, the numbers
would be measuring the constraints instead of the criterion, and the whole
comparison would be worthless.

Who is being compared
---------------------
========  ==================================================  ============
Code      Objective                                           Stands for
========  ==================================================  ============
B1        ``min ||z||_1``                                     ordering as
                                                              usual
B2        ``min ||z - shift(z_B1)||_1``                        the naive
                                                              fix: same
                                                              order, one
                                                              block sooner
B3        ``max sum_i r_i``                                    a forecast
                                                              correction
                                                              with no
                                                              fairness in
                                                              it
B4        lexicographic max-min, then least gate movement      this study
B5        ``min sigma(r)``                                     the nearest
                                                              published
                                                              fairness
                                                              criterion
========  ==================================================  ============

Why B1 and B2 are projections
-----------------------------
When the source is short, "change nothing" is not an option that exists:
the users' usual draws add up to more than the canal is given, so the
zero deviation is outside the feasible set. Both naive baselines are
therefore read as the *nearest feasible schedule* to what they would have
liked to do, in the one-norm. That is the most generous reading available
to them - it hands them the best feasible version of their own idea - and
it tilts every comparison here against this study rather than for it.

The time shift, and the honest thing to say about it
----------------------------------------------------
Shifting an order that is flat in time does nothing at all. In this
study's scenario the unchanged order *is* flat - every user asks for its
usual draw across the whole window - so bringing it forward by a block
returns it unchanged, and B2 read literally would be B1 with extra steps.
``test_a_flat_order_cannot_be_brought_forward`` measures that rather than
asserting it. What B2 shifts instead is B1's own answer: the naive
operator computes its schedule the usual way and then issues it a block
early to make up for the filter's group delay. That is a real, distinct
schedule, and it is still naive - nothing in it reasons about who ends up
worst off.

Standard deviation as a criterion
---------------------------------
B5 is Fan et al. (2023), *Water Resources Management* 37:1341-1365, whose
second objective is the standard deviation of the ratio of water actually
delivered to water demanded. That ratio is this study's ``r_i`` exactly;
the criterion applied to it is not. Minimising a spread can be done two
ways - by lifting the bottom, or by pulling the top down - and only the
first of those is what fairness is usually taken to mean. Whether the
second one actually happens on this canal is a question with a number for
an answer, which is why B5 is here and not merely discussed.

Its objective is quadratic and the feasible set is a polytope, so it is
solved by Frank-Wolfe: one linear programme per iteration over the very
same polytope, with an exact line search available in closed form because
the objective is a quadratic. Each iteration also re-mixes the schedules
found so far - a quadratic in as many variables as there are schedules,
which is free next to the linear programme and cuts the run from hundreds
of them to tens. The iteration count and the gap actually reached are
reported with the answer, never assumed, and the gap is a real bound: it
is computed from the schedule in hand and a vertex of the real polytope,
so it stays honest however the schedule was arrived at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

from faircanal.config import LP_METHOD, LP_OPTIONS
from faircanal.leximin import (
    LeximinError,
    RatioProgramme,
    refine,
    solve_leximin,
    solve_utilitarian,
)
from faircanal.plant import CanalPlant
from faircanal.programme import Programme, movement_tie_break
from faircanal.upperbound import free_gate_programme

__all__ = [
    "BaselineError",
    "BaselineResult",
    "CODES",
    "spread",
    "true_ratios",
    "brought_forward",
    "l1_nearest",
    "unchanged_order",
    "time_shift",
    "utilitarian",
    "leximin",
    "min_spread",
    "upper_bound",
    "solve_all",
]


class BaselineError(RuntimeError):
    """Raised when a baseline cannot be solved on this polytope.

    Every case means the point is infeasible or the solver failed, and
    both are answers the caller has to see: an infeasible point is where
    the certificate speaks, not where a number gets invented.
    """


#: The order the results table reports them in.
CODES = ("B1", "B2", "B3", "B4", "B5", "M1")

_NAMES = {
    "B1": "ordering as usual",
    "B2": "the same order, one block sooner",
    "B3": "utilitarian",
    "B4": "lexicographic max-min",
    "B5": "least spread of fractions",
    "M1": "free gates, an upper bound",
}


# ---------------------------------------------------------------------------
# Reading a schedule
# ---------------------------------------------------------------------------


def true_ratios(ratio: RatioProgramme, z: np.ndarray) -> np.ndarray:
    """What each user actually received, as a fraction of what it asked.

    The weighted fraction the lexicographic procedure maximises is this
    divided by the user's priority weight, so a weight below one lifts it
    above the real thing. Fulfilment is always read from this one.
    """
    z = np.asarray(z, dtype=float)
    return np.asarray(ratio.ratio_rows @ z + ratio.ratio_offset, dtype=float).ravel()


def spread(ratios: np.ndarray) -> float:
    """Standard deviation of the fractions, as Fan et al. (2023) define it.

    Their equations (10) and (11) divide by the number of terms, not by
    one less than it, so this is the population standard deviation and
    not the sample one. The difference is small and the point of quoting
    a competitor's number is that it is theirs.
    """
    values = np.asarray(ratios, dtype=float).ravel()
    if values.size == 0:
        raise BaselineError("a spread over no users is not a number")
    return float(np.sqrt(np.mean((values - values.mean()) ** 2)))


@dataclass(frozen=True)
class BaselineResult:
    """One alternative, solved, with everything needed to report it."""

    code: str
    name: str
    z: np.ndarray
    ratios: np.ndarray
    weighted: np.ndarray
    n_programmes: int
    objective: float
    detail: dict = field(default_factory=dict)

    @property
    def worst(self) -> float:
        return float(self.ratios.min())

    @property
    def total(self) -> float:
        return float(self.ratios.sum())

    @property
    def spread(self) -> float:
        return spread(self.ratios)

    @property
    def fulfilled(self) -> bool:
        return bool(np.all(self.ratios >= 1.0 - 1.0e-6))

    def line(self) -> str:
        return (
            f"{self.code} {self.name}: worst {self.worst:.4f}, total "
            f"{self.total:.4f}, spread {self.spread:.4f}, "
            f"{self.n_programmes} programmes"
        )


def _result(
    code: str,
    ratio: RatioProgramme,
    z: np.ndarray,
    n_programmes: int,
    objective: float,
    detail: dict | None = None,
) -> BaselineResult:
    z = np.asarray(z, dtype=float)
    return BaselineResult(
        code=code,
        name=_NAMES[code],
        z=z,
        ratios=true_ratios(ratio, z),
        weighted=ratio.ratios_at(z),
        n_programmes=n_programmes,
        objective=float(objective),
        detail=dict(detail or {}),
    )


# ---------------------------------------------------------------------------
# One linear programme, with the frozen settings
# ---------------------------------------------------------------------------


def _linprog(cost, a_ub, b_ub, bounds, a_eq, b_eq, what: str):
    solution = linprog(
        c=cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=LP_METHOD,
        options=dict(LP_OPTIONS),
    )
    if not solution.success:
        raise BaselineError(
            f"{what}: the solver returned status {solution.status} "
            f"({solution.message.strip()})"
        )
    return solution


def _widen(ratio: RatioProgramme, extra: int):
    """The same programme with *extra* auxiliary columns appended."""
    pad_ub = sparse.csr_matrix((ratio.a_ub.shape[0], extra))
    a_ub = sparse.hstack([ratio.a_ub, pad_ub], format="csr")
    a_eq = b_eq = None
    if ratio.a_eq is not None:
        a_eq = sparse.hstack(
            [ratio.a_eq, sparse.csr_matrix((ratio.a_eq.shape[0], extra))], format="csr"
        )
        b_eq = ratio.b_eq
    return a_ub, ratio.b_ub, a_eq, b_eq


# ---------------------------------------------------------------------------
# B1 and B2: the nearest feasible schedule to one somebody wanted
# ---------------------------------------------------------------------------


def l1_nearest(
    ratio: RatioProgramme, target: np.ndarray | None = None, what: str = "projection"
) -> tuple[np.ndarray, float, int]:
    """The feasible schedule closest to *target* in the one-norm.

    The absolute value is linearised the usual way: one auxiliary column
    per variable, held above both ``z - target`` and ``target - z``, with
    the cost pushing it down. The one-norm rather than the two-norm
    because it keeps the whole comparison inside one linear solver, and
    because it is the reading that leaves a schedule alone wherever it can
    be left alone instead of spreading a small correction over every
    block.
    """
    n = ratio.n_vars
    goal = np.zeros(n) if target is None else np.asarray(target, dtype=float).ravel()
    if goal.size != n:
        raise BaselineError(
            f"the target has {goal.size} entries but the programme has {n} variables"
        )

    eye = sparse.identity(n, format="csr")
    base_ub, base_b, a_eq, b_eq = _widen(ratio, n)
    a_ub = sparse.vstack(
        [
            base_ub,
            sparse.hstack([eye, -eye]),
            sparse.hstack([-eye, -eye]),
        ],
        format="csr",
    )
    b_ub = np.concatenate([base_b, goal, -goal])
    cost = np.concatenate([np.zeros(n), np.ones(n)])
    bounds = (*ratio.bounds, *(((0.0, None),) * n))

    solution = _linprog(cost, a_ub, b_ub, bounds, a_eq, b_eq, what)
    return np.asarray(solution.x[:n], dtype=float), float(solution.fun), 1


def brought_forward(
    schedule: np.ndarray, n_users: int, blocks: int, blocks_early: int = 1
) -> np.ndarray:
    """Issue a schedule *blocks_early* blocks sooner than it was written.

    What falls off the end is filled with zero rather than repeated,
    because past the last block the deviation *is* zero by construction -
    every user is back to drawing its usual amount - so zero is what the
    schedule already says about that time, not an assumption added here.
    """
    if blocks_early < 0:
        raise BaselineError(f"a shift of {blocks_early} blocks goes backwards")
    table = np.asarray(schedule, dtype=float).reshape(n_users, blocks)
    shifted = np.zeros_like(table)
    if blocks_early < blocks:
        shifted[:, : blocks - blocks_early] = table[:, blocks_early:]
    return shifted.ravel()


def unchanged_order(programme: Programme) -> BaselineResult:
    """B1: the feasible schedule nearest to ordering exactly as usual."""
    z, objective, solved = l1_nearest(programme.ratio, None, "B1 unchanged order")
    return _result(
        "B1",
        programme.ratio,
        z,
        solved,
        objective,
        {"norm": "L1", "target": "zero deviation"},
    )


def time_shift(
    programme: Programme, base: BaselineResult | None = None, blocks_early: int = 1
) -> BaselineResult:
    """B2: B1's schedule, issued a block early, then made feasible again.

    The block is not a free parameter. The filter's group delay is 11.08
    samples at one minute a sample, and an order block is fifteen minutes,
    so one block is the smallest shift an operator working in blocks can
    make, and the closest one to the delay being compensated.
    """
    if base is None:
        base = unchanged_order(programme)
    if base.code != "B1":
        raise BaselineError(f"B2 shifts B1's schedule, not {base.code}'s")

    target = brought_forward(
        base.z, programme.scenario.n_users, programme.blocks, blocks_early
    )
    z, objective, solved = l1_nearest(programme.ratio, target, "B2 time shift")
    return _result(
        "B2",
        programme.ratio,
        z,
        base.n_programmes + solved,
        objective,
        {
            "blocks_early": blocks_early,
            "shifted": "B1",
            "distance_from_shifted": objective,
            "shift_changed_the_target": bool(
                np.linalg.norm(target - base.z, ord=1) > 1.0e-12
            ),
        },
    )


# ---------------------------------------------------------------------------
# B3 and B4: the two criteria being compared
# ---------------------------------------------------------------------------


def utilitarian(programme: Programme) -> BaselineResult:
    """B3: maximise the sum of the fractions and see who pays for it."""
    try:
        z = solve_utilitarian(programme.ratio)
    except LeximinError as error:
        raise BaselineError(f"B3: {error}") from error
    ratios = true_ratios(programme.ratio, z)
    return _result("B3", programme.ratio, z, 1, float(ratios.sum()))


def leximin(programme: Programme, tie_break: bool = True) -> BaselineResult:
    """B4: this study - lexicographic max-min, then the smoothest of those."""
    try:
        result = solve_leximin(programme.ratio)
        if tie_break:
            result = refine(programme.ratio, result, movement_tie_break(programme))
    except LeximinError as error:
        raise BaselineError(f"B4: {error}") from error

    ratios = true_ratios(programme.ratio, result.z)
    return _result(
        "B4",
        programme.ratio,
        result.z,
        result.n_programmes,
        float(ratios.min()),
        {
            "stage_levels": result.stage_levels,
            "stage_saturated": result.stage_saturated,
            "accuracy_bound": result.accuracy_bound,
            "refined": result.refined,
            "tie_break_cost": result.tie_break_cost,
        },
    )


# ---------------------------------------------------------------------------
# M1: the bound, on its own programme
# ---------------------------------------------------------------------------


def upper_bound(programme: Programme, plant: CanalPlant) -> BaselineResult:
    """M1: what the canal could deliver with a perfect control layer.

    Not a competitor and never reported as one. It frees the gate
    commands, keeping every physical limit and the filter, so the gap
    between it and this study's answer is how much of the shortfall the
    control layer is responsible for. Because the controller's own
    commands are feasible here, its optimum is never below B4's, and
    ``test_the_upper_bound_is_above_the_proposed_method`` holds it to
    that.
    """
    free = free_gate_programme(programme, plant)
    try:
        result = solve_leximin(free.ratio)
    except LeximinError as error:
        raise BaselineError(f"M1: {error}") from error

    orders = free.orders_of(result.z)
    ratios = true_ratios(programme.ratio, orders)
    return _result(
        "M1",
        programme.ratio,
        orders,
        result.n_programmes,
        float(ratios.min()),
        {
            "stage_levels": result.stage_levels,
            "stage_saturated": result.stage_saturated,
            "accuracy_bound": result.accuracy_bound,
            "n_vars": free.ratio.n_vars,
            "n_equalities": free.ratio.a_eq.shape[0],
            "controller": "removed; filter and every physical limit kept",
        },
    )


# ---------------------------------------------------------------------------
# B5: the competing fairness criterion
# ---------------------------------------------------------------------------


def _centre(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def _least_spread_mixture(
    images: np.ndarray, start: np.ndarray
) -> np.ndarray | None:
    """The mixture of known schedules whose fractions spread least.

    ``images[:, j]`` is the vector of fractions that schedule ``j``
    delivers. Because the fractions are affine in the schedule, mixing
    schedules mixes their fractions, so the whole problem shrinks to a
    tiny quadratic over the simplex - one variable per schedule found so
    far, and only as many rows as there are users. Solving it is what
    makes the outer iteration finish in tens of linear programmes instead
    of hundreds.

    Returns ``None`` if the small solve fails, which the caller answers by
    taking an ordinary step instead. Nothing about correctness rests on
    this function: the gap that decides when to stop is computed from the
    schedule itself and from a vertex of the real polytope, so a bad
    mixture costs iterations and never accuracy.
    """
    centred = images - images.mean(axis=0, keepdims=True)
    hessian = centred.T @ centred / images.shape[0]
    size = images.shape[1]

    solution = minimize(
        lambda weights: float(weights @ hessian @ weights),
        start,
        jac=lambda weights: 2.0 * (hessian @ weights),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * size,
        constraints=[
            {
                "type": "eq",
                "fun": lambda weights: float(weights.sum() - 1.0),
                "jac": lambda weights: np.ones(size),
            }
        ],
        options={"maxiter": 200, "ftol": 1.0e-16},
    )
    if not solution.success:
        return None
    weights = np.clip(np.asarray(solution.x, dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return None
    return weights / total


def min_spread(
    programme: Programme,
    start: np.ndarray | None = None,
    max_iterations: int = 200,
    sigma_tol: float = 1.0e-6,
    corrective: bool = True,
) -> BaselineResult:
    """B5: minimise the standard deviation of the fractions.

    Frank-Wolfe, because the objective is quadratic and the feasible set
    is the polytope every other baseline uses: each iteration minimises a
    linear function over that polytope - one call to the same solver -
    and the gap that falls out is a genuine bound on how much of the
    objective is left to win, computed from the schedule in hand and a
    vertex of the real polytope.

    The plain method is correct here but slow - a few hundred linear
    programmes to settle - so each iteration also re-mixes every schedule
    found so far, which is the fully corrective variant. That inner
    problem is small enough to be free: the fractions are affine in the
    schedule, so mixing schedules mixes their fractions, and the whole
    thing becomes a quadratic in as many variables as there are schedules
    over as many rows as there are users. If it ever fails the iteration
    falls back to the ordinary step with its exact line search, which is
    available in closed form because the objective is a quadratic along
    any line.

    Squared spread is minimised rather than spread itself - the square
    root is increasing, so the minimiser is the same, and the square is
    differentiable at the point where every fraction is equal, which is
    exactly the point the method is trying to reach.

    Stopping, in the units the answer is reported in
    ------------------------------------------------
    A gap in squared spread is not a quantity anybody can read, and its
    meaning changes completely between a canal where the fractions can be
    equalised and one where they cannot. Two facts turn it into one that
    can be read. The gap bounds what is left: ``f - f* <= gap``. And a
    spread is never negative: ``f* >= 0``. Together they place the best
    reachable spread inside

        sqrt(max(0, f - gap))  <=  sigma*  <=  sqrt(f),

    and the iteration stops when that interval is narrower than
    ``sigma_tol``. What is reported is therefore not "it converged" but
    "the least spread this canal allows is this number, to within this
    much", which is a statement that survives being quoted.

    The rule also fixes the case that made the plain gap test useless:
    where the fractions can be equalised the optimum is zero, the
    gradient vanishes with it, and chasing the gap down takes hundreds of
    iterations to add digits to a number that is already zero to seven
    places. Under the interval rule that canal stops in single figures of
    iterations, with the same answer.
    """
    if max_iterations < 1:
        raise BaselineError(f"{max_iterations} iterations is not a run")
    if sigma_tol <= 0.0:
        raise BaselineError(f"a tolerance of {sigma_tol} can never be met")

    ratio = programme.ratio
    n_users = ratio.n_users
    solved = 0

    if start is None:
        z, _, solved = l1_nearest(ratio, None, "B5 starting point")
    else:
        z = np.asarray(start, dtype=float).ravel()
        if z.size != ratio.n_vars:
            raise BaselineError("the starting point has the wrong number of variables")

    def squared(vector: np.ndarray) -> float:
        centred = _centre(true_ratios(ratio, vector))
        return float(centred @ centred) / n_users

    # The starting point is kept as a mixture component, so the schedule
    # can always be written as a mixture of what has been found and the
    # inner solve can never make things worse than they already are.
    schedules = [z.copy()]
    images = [true_ratios(ratio, z)]
    weights = np.array([1.0])

    gap = float("inf")
    lowest = 0.0
    iterations = 0
    corrections = 0
    for iterations in range(1, max_iterations + 1):
        here = squared(z)
        centred = _centre(true_ratios(ratio, z))
        gradient = (2.0 / n_users) * np.asarray(
            ratio.ratio_rows.T @ centred, dtype=float
        ).ravel()

        vertex = _linprog(
            gradient,
            ratio.a_ub,
            ratio.b_ub,
            ratio.bounds,
            ratio.a_eq,
            ratio.b_eq,
            "B5 Frank-Wolfe step",
        )
        solved += 1
        candidate = np.asarray(vertex.x, dtype=float)
        direction = candidate - z
        slope = float(gradient @ direction)
        gap = -slope
        lowest = float(np.sqrt(max(0.0, here - gap)))
        if float(np.sqrt(max(0.0, here))) - lowest <= sigma_tol:
            break

        mixed = mixture = None
        if corrective:
            schedules.append(candidate)
            images.append(true_ratios(ratio, candidate))
            mixture = _least_spread_mixture(
                np.column_stack(images), np.append(weights, 0.0)
            )
            if mixture is not None:
                mixed = np.column_stack(schedules) @ mixture
        if mixed is not None and squared(mixed) <= squared(z) + 1.0e-15:
            z, weights = mixed, mixture
            corrections += 1
            continue

        # Fall back to one ordinary step, and start the mixture again from
        # where that lands so the bookkeeping keeps describing z.
        along = _centre(np.asarray(ratio.ratio_rows @ direction, dtype=float).ravel())
        curvature = float(along @ along) / n_users
        step = 1.0 if curvature <= 0.0 else min(1.0, max(0.0, -slope / (2.0 * curvature)))
        if step <= 0.0:
            break
        z = z + step * direction
        schedules, images, weights = [z.copy()], [true_ratios(ratio, z)], np.array([1.0])

    ratios = true_ratios(ratio, z)
    reached = spread(ratios)
    return _result(
        "B5",
        ratio,
        z,
        solved,
        reached,
        {
            "iterations": iterations,
            "gap": gap,
            "sigma_bounds": (lowest, reached),
            "certified_width": reached - lowest,
            "converged": bool(reached - lowest <= sigma_tol),
            "sigma_tol": sigma_tol,
            "corrections": corrections,
            "schedules_mixed": len(schedules),
            "method": (
                "fully corrective Frank-Wolfe, exact line search as fallback"
                if corrective
                else "Frank-Wolfe, exact line search"
            ),
        },
    )


# ---------------------------------------------------------------------------
# All of them
# ---------------------------------------------------------------------------


def solve_all(
    programme: Programme,
    codes: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5"),
    plant: CanalPlant | None = None,
) -> dict[str, BaselineResult]:
    """Solve the named alternatives on one programme, sharing what they share.

    B2 needs B1's answer, so B1 is solved once and handed on rather than
    solved twice; B5 starts from it as well, which costs nothing and
    starts the iteration inside the feasible set. M1 is the exception: it
    is a different programme with different variables, so it needs the
    plant and is solved on its own. A baseline that cannot be solved
    raises, because at an infeasible point the answer is a certificate and
    not a row of numbers.
    """
    unknown = set(codes) - set(CODES)
    if unknown:
        raise BaselineError(f"unknown baseline codes: {sorted(unknown)}")
    if "M1" in codes and plant is None:
        raise BaselineError(
            "M1 frees the gate commands, so it needs the plant the programme "
            "was built from; pass plant=..."
        )

    answers: dict[str, BaselineResult] = {}
    first: BaselineResult | None = None
    if {"B1", "B2", "B5"} & set(codes):
        first = unchanged_order(programme)
    if "B1" in codes:
        answers["B1"] = first
    if "B2" in codes:
        answers["B2"] = time_shift(programme, first)
    if "B3" in codes:
        answers["B3"] = utilitarian(programme)
    if "B4" in codes:
        answers["B4"] = leximin(programme)
    if "B5" in codes:
        answers["B5"] = min_spread(programme, start=first.z)
    if "M1" in codes:
        answers["M1"] = upper_bound(programme, plant)
    return {code: answers[code] for code in CODES if code in answers}
