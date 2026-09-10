"""Permanent tests for the lexicographic solver. Never deleted.

The solver is exercised on small sharing problems whose lexicographic
answer can be worked out on paper, so a failure points at the solver rather
than at the canal model. The canal only makes the polytope bigger; it does
not change what the procedure has to do.

The comparison against the utilitarian objective is here rather than in the
experiment scripts because it is a property of the criterion, not of the
canal: on a scarce network, maximising the total buys the total by pushing
the worst-off user down. If that ever stops being true of this
implementation, the study's central claim has quietly broken.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from faircanal.leximin import (
    LeximinError,
    RatioProgramme,
    TieBreak,
    refine,
    solve_leximin,
    solve_utilitarian,
)


def sharing_programme(
    demands: list[float],
    capacity: float,
    user_caps: list[float | None] | None = None,
    weights: list[float] | None = None,
    free_variable: bool = False,
) -> RatioProgramme:
    """One shared source, one delivery fraction per user.

    Variable layout: ``[x_0 .. x_{n-1}, r_0 .. r_{n-1}]``, optionally
    followed by one variable that appears nowhere else - used to give the
    optimal face a flat direction so the tie-break stage has something to
    decide.

    The cap at one lives in the bounds on ``r``, where it belongs: it is a
    modelling decision of this study, not a property of the solver.
    """
    n = len(demands)
    n_vars = 2 * n + (1 if free_variable else 0)
    caps = user_caps or [None] * n

    rows, rhs = [], []

    total = np.zeros(n_vars)
    total[:n] = 1.0
    rows.append(total)
    rhs.append(capacity)

    for i, demand in enumerate(demands):
        row = np.zeros(n_vars)
        row[n + i] = 1.0
        row[i] = -1.0 / demand
        rows.append(row)
        rhs.append(0.0)

    bounds: list[tuple[float | None, float | None]] = [(0.0, caps[i]) for i in range(n)]
    bounds += [(0.0, 1.0)] * n
    if free_variable:
        bounds.append((0.0, 1.0))

    ratio_rows = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), np.arange(n, 2 * n))), shape=(n, n_vars)
    )

    return RatioProgramme(
        n_vars=n_vars,
        a_ub=sparse.csr_matrix(np.array(rows)),
        b_ub=np.array(rhs, dtype=float),
        ratio_rows=ratio_rows,
        ratio_offset=np.zeros(n),
        weights=np.array(weights or [1.0] * n, dtype=float),
        names=tuple(f"u{i + 1}" for i in range(n)),
        bounds=tuple(bounds),
    )


# ---------------------------------------------------------------------------
# The criterion itself
# ---------------------------------------------------------------------------


def test_equal_users_share_equally():
    """With equal demands and one scarce source, everybody gets the same."""
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0)
    result = solve_leximin(programme)
    assert result.ratios == pytest.approx([0.4, 0.4, 0.4], abs=1e-7)
    assert len(result.stage_levels) == 1, "one stage: everybody saturates together"
    assert result.stage_saturated == (("u1", "u2", "u3"),)


def test_a_capped_user_is_locked_and_the_rest_share_what_is_left():
    """The staged structure, on a problem whose answer is arithmetic.

    User 3 can take at most 2 units of a demand of 10, so its fraction is
    fixed at 0.2 whatever anybody else does. The other two then share the
    remaining 13 units of a capacity of 15, giving 0.65 each.
    """
    programme = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0]
    )
    result = solve_leximin(programme)
    assert result.ratios == pytest.approx([0.65, 0.65, 0.2], abs=1e-7)
    assert result.stage_levels == pytest.approx([0.2, 0.65], abs=1e-7)
    assert result.stage_saturated == (("u3",), ("u1", "u2"))


def test_the_worst_off_user_is_raised_at_the_cost_of_the_total():
    """The claim the study rests on, in its smallest form.

    A user with a small demand converts water into fraction more cheaply
    than one with a large demand, so a utilitarian objective spends the
    water there and leaves somebody at nothing. The lexicographic
    criterion refuses that trade: it takes a smaller total and gives the
    worst-off user a real share.
    """
    programme = sharing_programme([10.0, 10.0, 2.0], capacity=12.0)

    lex = solve_leximin(programme)
    util_ratios = programme.ratios_at(solve_utilitarian(programme))

    assert lex.ratios == pytest.approx([6 / 11, 6 / 11, 6 / 11], abs=1e-7)
    assert util_ratios.sum() == pytest.approx(2.0, abs=1e-7)

    assert lex.ratios.min() > util_ratios.min(), (
        "the lexicographic criterion must protect the worst-off user"
    )
    assert lex.ratios.sum() < util_ratios.sum(), (
        "and it must pay for that with the total, or it is not a trade-off"
    )


def test_over_supply_cannot_inflate_the_measure():
    """With water to spare every fraction is exactly one, and no more.

    Without the cap, a user could be handed extra water inside the window
    and report a fraction above one, which would reward over-delivery.
    """
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=10_000.0)
    result = solve_leximin(programme)
    assert result.ratios == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert result.ratios.max() <= 1.0 + 1e-9


def test_a_larger_weight_means_a_larger_share():
    """Weights are priorities, and the direction is the documented one.

    The criterion equalises ``r / w``, so a user with twice the weight
    ends with twice the fraction. Getting the direction backwards would
    silently penalise exactly the users a policy meant to protect.
    """
    programme = sharing_programme(
        [10.0, 10.0, 10.0], capacity=12.0, weights=[1.0, 1.0, 2.0]
    )
    result = solve_leximin(programme)
    ratios = programme.ratio_rows @ result.z
    assert ratios == pytest.approx([0.3, 0.3, 0.6], abs=1e-7)
    assert ratios[2] == pytest.approx(2.0 * ratios[0], abs=1e-7)


# ---------------------------------------------------------------------------
# Termination, determinism and honesty about accuracy
# ---------------------------------------------------------------------------


def test_the_procedure_terminates_in_at_most_one_stage_per_user():
    programme = sharing_programme(
        [10.0, 8.0, 6.0, 4.0],
        capacity=9.0,
        user_caps=[1.0, 2.0, None, None],
    )
    result = solve_leximin(programme)
    assert 1 <= len(result.stage_levels) <= programme.n_users
    assert sum(len(s) for s in result.stage_saturated) == programme.n_users


def test_every_user_saturates_exactly_once():
    programme = sharing_programme(
        [10.0, 8.0, 6.0, 4.0], capacity=9.0, user_caps=[1.0, 2.0, None, None]
    )
    result = solve_leximin(programme)
    seen = [name for stage in result.stage_saturated for name in stage]
    assert sorted(seen) == sorted(programme.names)
    assert len(seen) == len(set(seen))


def test_stage_levels_do_not_decrease():
    """Later stages serve better-off users, so the level cannot fall."""
    programme = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0]
    )
    levels = solve_leximin(programme).stage_levels
    assert all(b >= a - 1e-9 for a, b in zip(levels, levels[1:]))


def test_two_runs_agree_byte_for_byte():
    """Determinism is a requirement, not a hope."""
    programme = sharing_programme(
        [10.0, 8.0, 6.0, 4.0], capacity=9.0, user_caps=[1.0, 2.0, None, None]
    )
    first, second = solve_leximin(programme), solve_leximin(programme)
    assert first.ratios.tobytes() == second.ratios.tobytes()
    assert first.stage_levels == second.stage_levels
    assert first.stage_saturated == second.stage_saturated


def test_the_accuracy_bound_is_reported():
    """The tolerance has a price and the result states it.

    Without this the levels would be called optimal, which they are not:
    a nearly saturated user can be locked one tolerance early at every
    stage.
    """
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0)
    result = solve_leximin(programme, eps_sat=1e-6)
    assert result.accuracy_bound == pytest.approx(3 * 1e-6)
    assert result.eps_sat == 1e-6


def test_marginals_are_returned_for_the_certificate():
    """The infeasibility certificate needs to say what is binding.

    On a problem limited by one source, the shadow price of that source
    must be non-zero; otherwise the certificate would report that nothing
    is in the way.
    """
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0)
    result = solve_leximin(programme)
    assert result.marginals.size >= programme.a_ub.shape[0]
    assert abs(result.marginals[0]) > 1e-9, "the shared source must price"


def test_the_number_of_programmes_solved_is_reported():
    """For the timing protocol: the cost is quadratic in the user count."""
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0)
    result = solve_leximin(programme)
    assert result.n_programmes == 1 + 3  # one stage plus one probe per user


# ---------------------------------------------------------------------------
# The tie-break stage
# ---------------------------------------------------------------------------


def flat_direction_tie_break(programme: RatioProgramme, sign: float) -> TieBreak:
    """Push the free variable to one end of its range."""
    cost = np.zeros(programme.n_vars)
    cost[-1] = sign
    return TieBreak(
        n_extra=0,
        a_ub=sparse.csr_matrix((0, programme.n_vars)),
        b_ub=np.zeros(0),
        cost=cost,
        bounds_extra=(),
    )


def test_the_tie_break_selects_within_the_optimal_face():
    """It decides what fairness leaves undecided, and nothing more.

    The free variable does not enter any constraint or any fraction, so
    the lexicographic optimum says nothing about it. Two opposite
    tie-break costs must therefore move it to opposite ends while leaving
    every fraction exactly where it was.
    """
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0, free_variable=True)
    base = solve_leximin(programme)

    low = refine(programme, base, flat_direction_tie_break(programme, +1.0))
    high = refine(programme, base, flat_direction_tie_break(programme, -1.0))

    assert low.z[-1] == pytest.approx(0.0, abs=1e-9)
    assert high.z[-1] == pytest.approx(1.0, abs=1e-9)

    assert low.ratios == pytest.approx(base.ratios, abs=1e-6)
    assert high.ratios == pytest.approx(base.ratios, abs=1e-6)
    assert low.refined and high.refined


def test_the_tie_break_is_itself_deterministic():
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0, free_variable=True)
    base = solve_leximin(programme)
    tie = flat_direction_tie_break(programme, +1.0)
    first, second = refine(programme, base, tie), refine(programme, base, tie)
    assert first.z.tobytes() == second.z.tobytes()
    assert first.tie_break_cost == second.tie_break_cost


def test_the_tie_break_keeps_the_stage_record():
    """The refinement reports the levels the staged procedure reached; it
    does not get to rewrite them."""
    programme = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0]
    )
    base = solve_leximin(programme)
    tie = TieBreak(
        n_extra=0,
        a_ub=sparse.csr_matrix((0, programme.n_vars)),
        b_ub=np.zeros(0),
        cost=np.zeros(programme.n_vars),
        bounds_extra=(),
    )
    refined = refine(programme, base, tie)
    assert refined.stage_levels == base.stage_levels
    assert refined.stage_saturated == base.stage_saturated


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_infeasible_programme_is_an_error_not_an_answer():
    programme = sharing_programme([10.0], capacity=-1.0)
    with pytest.raises(LeximinError, match="solver returned status"):
        solve_leximin(programme)


def test_a_zero_weight_is_refused():
    with pytest.raises(ValueError, match="strictly positive"):
        sharing_programme([10.0, 10.0], capacity=5.0, weights=[1.0, 0.0])


def test_duplicate_user_names_are_refused():
    programme = sharing_programme([10.0, 10.0], capacity=5.0)
    with pytest.raises(ValueError, match="unique"):
        RatioProgramme(
            n_vars=programme.n_vars,
            a_ub=programme.a_ub,
            b_ub=programme.b_ub,
            ratio_rows=programme.ratio_rows,
            ratio_offset=programme.ratio_offset,
            weights=programme.weights,
            names=("u1", "u1"),
            bounds=programme.bounds,
        )


def test_a_non_positive_tolerance_is_refused():
    programme = sharing_programme([10.0, 10.0], capacity=5.0)
    with pytest.raises(ValueError, match="eps_sat must be positive"):
        solve_leximin(programme, eps_sat=0.0)


def test_a_mis_shaped_ratio_matrix_is_refused():
    programme = sharing_programme([10.0, 10.0], capacity=5.0)
    with pytest.raises(ValueError, match="ratio_rows must be"):
        RatioProgramme(
            n_vars=programme.n_vars,
            a_ub=programme.a_ub,
            b_ub=programme.b_ub,
            ratio_rows=sparse.csr_matrix((3, programme.n_vars)),
            ratio_offset=np.zeros(3),
            weights=np.ones(2),
            names=programme.names,
            bounds=programme.bounds,
        )
