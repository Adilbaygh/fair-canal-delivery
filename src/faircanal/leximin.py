"""Lexicographic max-min over a polytope.

The fairness criterion of this study is not "maximise the total" and not
"equalise everybody". It is: raise the worst-off user's delivered fraction
as high as it will go; then, without giving any of that back, raise the
second worst-off; and so on. That ordering is what stops a method from
buying a large total by abandoning one user, which is exactly what a
utilitarian objective does when water is scarce.

Why this can be solved at all
-----------------------------
The delivered volume is affine in the reshaped order, and every constraint
is linear, so the feasible set is a polytope and each user's fraction is an
affine function on it. Under those two conditions - and only under them -
the staged procedure below terminates in at most one stage per user:

    at each stage, either the level found is the best the remaining users
    can all reach, or every one of them could be raised at once, and then
    their average could too, contradicting optimality.

That argument uses convexity of the feasible set and affinity of the
fractions. If a non-convex loss model were ever substituted, the argument
fails and the loop can run forever, so ``solve_leximin`` refuses to
continue rather than spinning when a stage saturates nobody.

Determinism
-----------
Two correct runs must produce the same bytes, so:

* the saturation test walks users in index order, never in set order;
* saturated users are recorded as sorted tuples;
* the saturation tolerance is a declared constant, three orders of
  magnitude above the solver's feasibility tolerance, so a user is never
  declared saturated because of solver noise;
* the solver, its options and its version are frozen elsewhere and
  recorded with every run.

The price of the tolerance is stated with the result: the levels returned
can fall short of the true lexicographic optimum by at most
``n_users * eps_sat``. That bound is reported, and the word "optimal" is
not used without it.

Uniqueness
----------
At the lexicographic optimum the vector of fractions is unique, but the
decision vector that produces it usually is not: many order schedules give
the same fairness outcome, and a solver returns whichever vertex it
reaches. ``refine`` runs one further stage that picks, among all
lexicographically optimal solutions, the one minimising a caller-supplied
linear cost - in this study, total gate movement, so the schedule chosen is
also the smoothest one. That reduces the ambiguity by orders of magnitude
but does not remove it, which is why the reported results are the
fractions and the objective values rather than the schedule itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from faircanal.config import EPS_SAT, LP_METHOD, LP_OPTIONS

__all__ = [
    "RatioProgramme",
    "TieBreak",
    "LeximinResult",
    "LeximinError",
    "solve_leximin",
    "refine",
    "solve_utilitarian",
]


class LeximinError(RuntimeError):
    """Raised when the staged procedure cannot proceed.

    Every case this covers means an assumption has been broken - an
    infeasible programme, a solver failure, or a stage that saturates
    nobody - and none of them may be papered over with a partial answer.
    """


@dataclass(frozen=True)
class RatioProgramme:
    """A polytope together with one affine fraction per user.

    The feasible set is

        a_ub @ z <= b_ub,   a_eq @ z == b_eq,   bounds[k] on z[k],

    and user ``i``'s weighted fraction is

        r_tilde[i] = (ratio_rows[i] @ z + ratio_offset[i]) / weights[i].

    The cap at one is not imposed here: it belongs in ``a_ub`` with the
    rest of the model, because a fraction that may exceed one is a
    modelling decision and not a property of this solver.
    """

    n_vars: int
    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    ratio_rows: sparse.csr_matrix
    ratio_offset: np.ndarray
    weights: np.ndarray
    names: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]
    a_eq: sparse.csr_matrix | None = None
    b_eq: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.a_ub.shape[1] != self.n_vars:
            raise ValueError(
                f"a_ub has {self.a_ub.shape[1]} columns but the programme has "
                f"{self.n_vars} variables"
            )
        if self.a_ub.shape[0] != self.b_ub.size:
            raise ValueError("a_ub and b_ub disagree on the number of rows")
        if self.ratio_rows.shape != (self.n_users, self.n_vars):
            raise ValueError(
                f"ratio_rows must be {(self.n_users, self.n_vars)}, "
                f"got {self.ratio_rows.shape}"
            )
        if self.ratio_offset.size != self.n_users:
            raise ValueError("ratio_offset must have one entry per user")
        if len(self.bounds) != self.n_vars:
            raise ValueError("bounds must have one entry per variable")
        if np.any(self.weights <= 0.0):
            raise ValueError(
                "weights must be strictly positive; a zero weight would make the "
                "fraction undefined rather than the user unimportant"
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"user names must be unique, got {self.names}")
        if (self.a_eq is None) != (self.b_eq is None):
            raise ValueError("a_eq and b_eq must be given together or not at all")

    @property
    def n_users(self) -> int:
        return len(self.names)

    def ratios_at(self, z: np.ndarray) -> np.ndarray:
        """Weighted fractions of a given decision vector."""
        return (self.ratio_rows @ z + self.ratio_offset) / self.weights


@dataclass(frozen=True)
class TieBreak:
    """A linear cost used to pick one lexicographically optimal solution.

    ``n_extra`` auxiliary variables are appended to the decision vector.
    ``a_ub`` and ``b_ub`` are stated over the extended vector ``[z, extra]``
    - this is how an absolute value is linearised - and ``cost`` is the
    objective over that same extended vector.
    """

    n_extra: int
    a_ub: sparse.csr_matrix
    b_ub: np.ndarray
    cost: np.ndarray
    bounds_extra: tuple[tuple[float | None, float | None], ...]

    def __post_init__(self) -> None:
        if len(self.bounds_extra) != self.n_extra:
            raise ValueError("bounds_extra must have one entry per auxiliary variable")
        if self.a_ub.shape[0] != self.b_ub.size:
            raise ValueError("a_ub and b_ub disagree on the number of rows")


@dataclass(frozen=True)
class LeximinResult:
    """Outcome of the staged procedure.

    Attributes
    ----------
    z:
        A decision vector attaining the lexicographic optimum. Not unique;
        see the module docstring.
    ratios:
        Weighted fractions, one per user, in the programme's user order.
    stage_levels:
        The level solved for at each stage, in order.
    stage_saturated:
        The users that saturated at each stage, as sorted tuples of names.
    accuracy_bound:
        ``n_users * eps_sat``: how far the reported levels may fall short
        of the true lexicographic optimum.
    marginals:
        Shadow prices of the inequality constraints at the final stage,
        for the infeasibility certificate. Non-positive by the solver's
        convention for a minimisation with ``a_ub @ z <= b_ub``.
    n_programmes:
        How many linear programmes were solved, for the timing protocol.
    """

    z: np.ndarray
    ratios: np.ndarray
    stage_levels: tuple[float, ...]
    stage_saturated: tuple[tuple[str, ...], ...]
    accuracy_bound: float
    marginals: np.ndarray
    n_programmes: int
    eps_sat: float
    refined: bool = False
    tie_break_cost: float | None = None
    solver: dict = field(default_factory=lambda: {"method": LP_METHOD, **dict(LP_OPTIONS)})


# ---------------------------------------------------------------------------
# Building and solving one linear programme
# ---------------------------------------------------------------------------


def _solve(
    cost: np.ndarray,
    a_ub: sparse.csr_matrix,
    b_ub: np.ndarray,
    bounds: tuple,
    a_eq: sparse.csr_matrix | None,
    b_eq: np.ndarray | None,
    what: str,
):
    """Solve one LP with the frozen settings, or say why it could not be."""
    result = linprog(
        c=cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method=LP_METHOD,
        options=dict(LP_OPTIONS),
    )
    if not result.success:
        raise LeximinError(
            f"{what}: the solver returned status {result.status} "
            f"({result.message.strip()})"
        )
    return result


def _lower_bound_rows(
    programme: RatioProgramme, users: list[int], levels: np.ndarray, n_pad: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Rows enforcing ``r_tilde[i] >= levels[i]`` for the listed users.

    Written as ``-(ratio_rows[i] / w_i) @ z <= offset_i / w_i - level_i``
    so that everything stays in the solver's ``a_ub @ x <= b_ub`` form.
    ``n_pad`` zero columns are appended for auxiliary variables.
    """
    if not users:
        return sparse.csr_matrix((0, programme.n_vars + n_pad)), np.zeros(0)

    scale = sparse.diags(1.0 / programme.weights[users])
    rows = -(scale @ programme.ratio_rows[users, :])
    rhs = programme.ratio_offset[users] / programme.weights[users] - levels
    if n_pad:
        rows = sparse.hstack([rows, sparse.csr_matrix((len(users), n_pad))])
    return sparse.csr_matrix(rows), rhs


def _stage_programme(
    programme: RatioProgramme, active: list[int], locked: dict[int, float]
):
    """Maximise a common level for the active users, respecting locked ones.

    One auxiliary column carries the level itself, so the constraint
    ``r_tilde[i] >= lambda`` becomes a linear row like any other.
    """
    n = programme.n_vars
    pad = sparse.csr_matrix((programme.a_ub.shape[0], 1))
    blocks = [sparse.hstack([programme.a_ub, pad])]
    rhs = [programme.b_ub]

    # r_tilde[i] - lambda >= 0  for the active users
    scale = sparse.diags(1.0 / programme.weights[active])
    active_rows = sparse.hstack(
        [-(scale @ programme.ratio_rows[active, :]), np.ones((len(active), 1))]
    )
    blocks.append(sparse.csr_matrix(active_rows))
    rhs.append(programme.ratio_offset[active] / programme.weights[active])

    if locked:
        users = sorted(locked)
        rows, values = _lower_bound_rows(
            programme, users, np.array([locked[i] for i in users]), n_pad=1
        )
        blocks.append(rows)
        rhs.append(values)

    a_ub = sparse.vstack(blocks, format="csr")
    b_ub = np.concatenate(rhs)

    cost = np.zeros(n + 1)
    cost[-1] = -1.0  # maximise the level
    bounds = (*programme.bounds, (None, None))

    a_eq = b_eq = None
    if programme.a_eq is not None:
        a_eq = sparse.hstack(
            [programme.a_eq, sparse.csr_matrix((programme.a_eq.shape[0], 1))],
            format="csr",
        )
        b_eq = programme.b_eq

    return cost, a_ub, b_ub, bounds, a_eq, b_eq


def _saturation_programme(
    programme: RatioProgramme, user: int, floors: dict[int, float]
):
    """Maximise one user's fraction while nobody drops below their floor."""
    users = sorted(floors)
    rows, values = _lower_bound_rows(
        programme, users, np.array([floors[i] for i in users]), n_pad=0
    )
    a_ub = sparse.vstack([programme.a_ub, rows], format="csr")
    b_ub = np.concatenate([programme.b_ub, values])

    cost = -np.asarray(
        programme.ratio_rows[[user], :].todense()
    ).ravel() / programme.weights[user]

    return cost, a_ub, b_ub, programme.bounds, programme.a_eq, programme.b_eq


# ---------------------------------------------------------------------------
# The staged procedure
# ---------------------------------------------------------------------------


def solve_leximin(
    programme: RatioProgramme, eps_sat: float = EPS_SAT
) -> LeximinResult:
    """Lexicographically maximise the vector of weighted fractions.

    Raises
    ------
    LeximinError
        If the programme is infeasible, the solver fails, or a stage
        saturates nobody. The last of these means the convexity or
        affinity assumption has been broken, and continuing would loop
        forever.
    """
    if eps_sat <= 0.0:
        raise ValueError(f"eps_sat must be positive, got {eps_sat}")

    active = list(range(programme.n_users))
    locked: dict[int, float] = {}
    levels: list[float] = []
    saturated_per_stage: list[tuple[str, ...]] = []
    solved = 0
    last_stage_result = None

    while active:
        if len(levels) > programme.n_users:
            raise LeximinError(
                "more stages than users; the staged procedure is not terminating"
            )

        stage = _solve(*_stage_programme(programme, active, locked), what="stage")
        solved += 1
        last_stage_result = stage
        level = float(stage.x[-1])
        levels.append(level)

        floors = {**locked, **{i: level for i in active}}
        saturated: list[int] = []
        for user in active:  # index order, never set order
            probe = _solve(
                *_saturation_programme(programme, user, floors), what="saturation test"
            )
            solved += 1
            reachable = -float(probe.fun)
            if reachable <= level + eps_sat:
                saturated.append(user)

        if not saturated:
            raise LeximinError(
                f"stage {len(levels)} saturated nobody at level {level}. Every user "
                f"could be raised at once, so the level was not optimal - which can "
                f"only happen if the feasible set is not convex or a fraction is not "
                f"affine. Refusing to loop."
            )

        for user in saturated:
            locked[user] = level
            active.remove(user)
        saturated_per_stage.append(tuple(programme.names[i] for i in sorted(saturated)))

    z = np.asarray(last_stage_result.x[:-1], dtype=float)
    return LeximinResult(
        z=z,
        ratios=programme.ratios_at(z),
        stage_levels=tuple(levels),
        stage_saturated=tuple(saturated_per_stage),
        accuracy_bound=programme.n_users * eps_sat,
        marginals=np.asarray(last_stage_result.ineqlin.marginals, dtype=float),
        n_programmes=solved,
        eps_sat=eps_sat,
    )


def refine(
    programme: RatioProgramme, result: LeximinResult, tie_break: TieBreak
) -> LeximinResult:
    """Pick one lexicographically optimal solution by a secondary cost.

    The fractions are held at the levels the staged procedure reached -
    with one saturation tolerance of slack, so the refinement cannot be
    made infeasible by the same rounding that produced those levels - and
    the caller's linear cost is minimised over what remains.
    """
    n, m = programme.n_vars, tie_break.n_extra
    users = list(range(programme.n_users))
    floors = result.ratios - result.eps_sat

    rows, values = _lower_bound_rows(programme, users, floors, n_pad=m)
    base = sparse.hstack(
        [programme.a_ub, sparse.csr_matrix((programme.a_ub.shape[0], m))], format="csr"
    )
    a_ub = sparse.vstack([base, rows, tie_break.a_ub], format="csr")
    b_ub = np.concatenate([programme.b_ub, values, tie_break.b_ub])

    a_eq = b_eq = None
    if programme.a_eq is not None:
        a_eq = sparse.hstack(
            [programme.a_eq, sparse.csr_matrix((programme.a_eq.shape[0], m))],
            format="csr",
        )
        b_eq = programme.b_eq

    bounds = (*programme.bounds, *tie_break.bounds_extra)
    solution = _solve(
        tie_break.cost, a_ub, b_ub, bounds, a_eq, b_eq, what="tie-break stage"
    )

    z = np.asarray(solution.x[:n], dtype=float)
    return LeximinResult(
        z=z,
        ratios=programme.ratios_at(z),
        stage_levels=result.stage_levels,
        stage_saturated=result.stage_saturated,
        accuracy_bound=result.accuracy_bound,
        marginals=np.asarray(solution.ineqlin.marginals, dtype=float),
        n_programmes=result.n_programmes + 1,
        eps_sat=result.eps_sat,
        refined=True,
        tie_break_cost=float(solution.fun),
    )


def solve_utilitarian(programme: RatioProgramme) -> np.ndarray:
    """Maximise the sum of the weighted fractions.

    Provided as a baseline, not as an alternative worth recommending: on a
    scarce network it buys a larger total by pushing the worst-off user
    down, which is the comparison the study exists to make.
    """
    scale = sparse.diags(1.0 / programme.weights)
    cost = -np.asarray((scale @ programme.ratio_rows).sum(axis=0)).ravel()
    solution = _solve(
        cost,
        programme.a_ub,
        programme.b_ub,
        programme.bounds,
        programme.a_eq,
        programme.b_eq,
        what="utilitarian baseline",
    )
    return np.asarray(solution.x, dtype=float)
