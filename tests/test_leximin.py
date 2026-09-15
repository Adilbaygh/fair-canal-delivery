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

from faircanal.config import LP_METHOD
from faircanal.leximin import (
    SolverFailure,
    SolverLimit,
    SolverUndecided,
    undecided_kind,
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
    offset: float = 0.0,
) -> RatioProgramme:
    """One shared source, one delivery fraction per user.

    Variable layout: ``[x_0 .. x_{n-1}, r_0 .. r_{n-1}]``, optionally
    followed by one variable that appears nowhere else - used to give the
    optimal face a flat direction so the tie-break stage has something to
    decide.

    The cap at one lives in the bounds on ``r``, where it belongs: it is a
    modelling decision of this study, not a property of the solver.

    ``offset`` writes part of each fraction outside the decision vector.
    The stored variable becomes ``r_i - offset`` and the programme declares
    ``ratio_offset = offset``, with the bound and the coupling row moved by
    the same amount, so the feasible set and every fraction are unchanged -
    only the bookkeeping differs. The canal's own programme is written this
    way, in deviations from nominal operation, and nothing that reads a
    fraction may notice the difference.
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
        rhs.append(-offset)

    bounds: list[tuple[float | None, float | None]] = [(0.0, caps[i]) for i in range(n)]
    bounds += [(0.0 - offset, 1.0 - offset)] * n
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
        ratio_offset=np.full(n, offset),
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


def test_the_answer_does_not_depend_on_where_the_constant_is_written():
    """The same fraction, split differently between row and offset.

    A fraction is ``(ratio_rows @ z + ratio_offset) / w``. Nothing says
    how much of a constant term lives in the row and how much in the
    offset: shifting a constant from one to the other and compensating in
    the bound describes the same problem, so it must give the same
    answer.

    It did not. The saturation test maximised the row alone and compared
    the result against a level that meant the whole fraction, so on any
    programme with a non-zero offset every user was declared saturated at
    the first stage. The procedure stopped there and returned max-min
    where it promised leximin - the worst-off level correct, every level
    above it whatever vertex the solver happened to return. Every
    programme in this file had a zero offset, which is why nothing failed;
    the canal's own programme is written in deviations from nominal
    operation, where the offset is one.
    """
    plain = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0]
    )
    shifted = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0], offset=1.0
    )

    first, second = solve_leximin(plain), solve_leximin(shifted)
    assert second.ratios == pytest.approx(first.ratios, abs=1e-7)
    assert second.stage_levels == pytest.approx(first.stage_levels, abs=1e-7)
    assert second.stage_saturated == first.stage_saturated
    assert len(second.stage_levels) == 2, (
        "the offset collapsed the procedure to a single stage, which is "
        "max-min and not leximin"
    )


def test_a_user_that_can_still_be_raised_is_not_declared_saturated():
    """The bug in its smallest form, stated as the property it broke.

    A user sitting strictly above the common level, in a point the
    procedure itself returned, cannot be saturated at that level: the
    point is the witness. This checks the record rather than the internals
    - every user the procedure locks at a stage must be one that the
    returned schedule leaves at that stage's level, to within the
    tolerance.
    """
    shifted = sharing_programme(
        [10.0, 8.0, 6.0, 4.0],
        capacity=9.0,
        user_caps=[1.0, 2.0, None, None],
        offset=1.0,
    )
    result = solve_leximin(shifted)
    index = {name: i for i, name in enumerate(shifted.names)}
    assert len(result.stage_levels) >= 2, (
        "the offset collapsed the procedure to a single stage, which is "
        "max-min and not leximin"
    )
    for level, names in zip(result.stage_levels, result.stage_saturated):
        for name in names:
            assert result.ratios[index[name]] == pytest.approx(level, abs=1e-5), (
                f"{name} was locked at {level} but the returned schedule "
                f"leaves it at {result.ratios[index[name]]}"
            )


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


def test_the_prices_answer_for_the_worst_off_user_not_the_best_off():
    """Whose objective the shadow prices belong to.

    Three users share 15 units and user 3 may take at most 2, so the
    worst-off level is 0.2 and it is that user's own cap, not the shared
    source, that sets it: at the first stage only 6 of the 15 units are
    spoken for. The second stage then shares what is left and empties the
    source.

    So the source is tight at the schedule that comes back, and its price
    is nevertheless zero - a unit more source would not lift the worst-off
    user at all. That is the property the certificate needs, because its
    sentence is about the worst-off user. Built from the last stage's
    duals it would name the source as what stands in the way of a user the
    source cannot help; built from the tie-break's it would report the
    price of gate movement in units of flow.
    """
    programme = sharing_programme(
        [10.0, 10.0, 10.0], capacity=15.0, user_caps=[None, None, 2.0],
        free_variable=True,
    )
    result = solve_leximin(programme)
    assert len(result.stage_levels) == 2
    assert result.z[:3].sum() == pytest.approx(15.0, abs=1e-7), (
        "the second stage should empty the source; if it does not, this "
        "test no longer distinguishes the two sets of duals"
    )
    assert abs(result.marginals[0]) <= 1e-9, (
        "the shared source is priced although it does not hold the "
        "worst-off user down: these are a later stage's duals"
    )

    refined = refine(programme, result, flat_direction_tie_break(programme, +1.0))
    assert refined.marginals == pytest.approx(result.marginals), (
        "the tie-break's duals price gate movement, not water, and must "
        "not replace the first stage's"
    )


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


def test_the_tie_break_costs_one_more_tolerance_and_says_so():
    """What is reported is the refined schedule, so the bound is its own.

    The tie-break holds each user at the level the stages reached *less*
    one saturation tolerance - otherwise the same rounding that produced
    those levels could make the refinement infeasible - so a schedule that
    has been through it can be one tolerance further from the exact
    optimum than the staged procedure's own bound admits. That extra
    tolerance was not carried, and the bound went into every result file
    and into the article one tolerance too small.
    """
    programme = sharing_programme([10.0, 10.0, 10.0], capacity=12.0)
    staged = solve_leximin(programme, eps_sat=1e-6)
    refined = refine(programme, staged, flat_direction_tie_break(programme, +1.0))
    assert refined.refined
    assert refined.accuracy_bound == pytest.approx(staged.accuracy_bound + 1e-6)
    assert refined.accuracy_bound == pytest.approx(
        (programme.n_users + 1) * 1e-6
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


# ---------------------------------------------------------------------------
# A limit is not a verdict
# ---------------------------------------------------------------------------


def test_the_two_failures_cannot_be_caught_by_accident():
    """The distinction a two-hour stall taught us to make explicit.

    Everything upstream answers a LeximinError with an infeasibility
    certificate - "no schedule exists on these terms, however the water is
    shared". If a solver that was told to stop after five minutes raised
    that same exception, that sentence would be written about a canal
    nobody asked about. So the limit has its own exception, and it is
    deliberately outside the LeximinError branch of the tree: catching it
    has to be done on purpose.

    A future refactor that made one a subclass of the other would turn
    every timeout into a claim about the canal, and nothing else in the
    suite would notice. This notices.
    """
    assert not issubclass(SolverUndecided, LeximinError)
    assert not issubclass(LeximinError, SolverUndecided)
    assert issubclass(SolverLimit, SolverUndecided)
    assert issubclass(SolverFailure, SolverUndecided)


def test_a_solver_that_stops_on_a_limit_raises_the_limit_not_the_error(monkeypatch):
    """Status one is a limit, and it is mapped to the exception that says so.

    Driven through a stand-in rather than a real timeout, because a real
    one depends on how fast the machine is, and a test that passes on a
    slow machine and not a fast one is not a test.
    """
    from scipy.optimize import OptimizeResult

    from faircanal import leximin as module

    def stopped(*args, **kwargs):
        return OptimizeResult(
            x=None, fun=None, success=False, status=1,
            message="Time limit reached", nit=0,
        )

    monkeypatch.setattr(module, "linprog", stopped)
    with pytest.raises(SolverLimit, match="not infeasibility"):
        solve_leximin(sharing_programme([1.0, 1.0], capacity=1.0))


@pytest.mark.parametrize(
    "status, expected",
    [(1, SolverLimit), (3, SolverFailure), (4, SolverFailure)],
)
def test_only_one_status_means_infeasible(monkeypatch, status, expected):
    """The mistake that got through the first guard, closed for good.

    The first version of this guard caught the time limit and nothing
    else, so when the solver came back with status four - numerical
    trouble on a large degenerate programme - the run wrote down
    "infeasible" for three scan points. Status four is the solver saying
    it ran out of arithmetic, not the canal saying it has no schedule.

    Exactly one status, two, is a statement about the problem. Every other
    way of not returning an answer is a statement about the solve, and
    each of them is checked here by name.
    """
    from scipy.optimize import OptimizeResult

    from faircanal import leximin as module

    def undecided(*args, **kwargs):
        return OptimizeResult(
            x=None, fun=None, success=False, status=status,
            message=f"stand-in for status {status}", nit=0,
        )

    monkeypatch.setattr(module, "linprog", undecided)
    with pytest.raises(expected) as raised:
        solve_leximin(sharing_programme([1.0, 1.0], capacity=1.0))
    assert raised.value.status == status
    assert not isinstance(raised.value, LeximinError)


def test_an_infeasible_programme_is_still_an_infeasible_programme(monkeypatch):
    """The other half: status two must keep meaning what it means."""
    from scipy.optimize import OptimizeResult

    from faircanal import leximin as module

    def infeasible(*args, **kwargs):
        return OptimizeResult(
            x=None, fun=None, success=False, status=2,
            message="The problem is infeasible.", nit=0,
        )

    monkeypatch.setattr(module, "linprog", infeasible)
    with pytest.raises(LeximinError):
        solve_leximin(sharing_programme([1.0, 1.0], capacity=1.0))


def test_a_time_limit_can_be_asked_for_without_changing_the_answer():
    """The override is additive: same programme, same fractions."""
    programme = sharing_programme([1.0, 1.0, 1.0], capacity=2.0)
    frozen = solve_leximin(programme)
    generous = solve_leximin(programme, options={"time_limit": 600.0})
    assert np.allclose(frozen.ratios, generous.ratios)


def test_a_verdict_hidden_in_the_message_is_still_a_verdict():
    """The scipy wrapper can hide an answer inside a status that denies one.

    On scipy 1.18 a genuinely infeasible programme comes back as status 4,
    "numerical difficulties", because the wrapper does not recognise the
    HiGHS status that carried the verdict. The verdict is still in the
    message: ``model_status is Unknown; primal_status is Infeasible``.
    Reading only the code turned four real infeasibilities into "the
    solver gave up", which is how this test came to exist.

    The exact strings this solver produced are pinned here, so a future
    version that reworded them fails loudly instead of quietly changing
    what the study reports.
    """
    hidden = (
        "The HiGHS status code was not recognized. (HiGHS Status 15: "
        "model_status is Unknown; primal_status is Infeasible)"
    )
    assert undecided_kind(4, hidden) is None

    # And the ones that really are the solver giving up stay that way.
    assert undecided_kind(4, "(HiGHS Status 4: Solve error)") is SolverFailure
    assert undecided_kind(4, "(HiGHS Status 0: Not Set)") is SolverFailure
    assert undecided_kind(1, "Time limit reached") is SolverLimit
    assert undecided_kind(3, "The problem is unbounded.") is SolverFailure

    # Status two never needed interpreting and still does not.
    assert undecided_kind(2, "The problem is infeasible.") is None
    assert undecided_kind(0, "Optimization terminated successfully.") is None


def test_the_reading_errs_towards_rerunning_rather_than_accusing_the_canal():
    """Which way the doubt falls, stated as a test.

    Reading a real infeasibility as a solver failure costs a rerun.
    Reading a solver failure as a real infeasibility puts a false
    statement about the canal into a certificate. So only a message that
    says infeasible in as many words flips a non-deciding status, and
    anything vaguer stays undecided.
    """
    for vague in ("model_status is Unknown", "primal_status is None", ""):
        assert undecided_kind(4, vague) is SolverFailure


# ---------------------------------------------------------------------------
# Asking a second algorithm before recording that nobody decided
# ---------------------------------------------------------------------------


def _fake_linprog(answers):
    """A stand-in solver whose answer depends on which algorithm is asked.

    Driven through a stand-in for the same reason as the limit above: the
    real programme that separates these two algorithms takes fifty
    seconds to fail on one machine and half a second to succeed on
    another, and a test that depends on which machine it runs on is not
    a test. The behaviour being pinned is the dispatch, not the pivoting.
    """
    from scipy.optimize import OptimizeResult

    asked = []

    def call(*args, method=None, **kwargs):
        asked.append(method)
        status, message = answers[method]
        return OptimizeResult(
            x=np.zeros(4), fun=0.0, success=(status == 0), status=status,
            message=message, nit=0,
            ineqlin=OptimizeResult(marginals=np.zeros(1)),
            eqlin=OptimizeResult(marginals=np.zeros(0)),
        )

    call.asked = asked
    return call


def test_a_second_algorithm_is_asked_before_recording_that_nobody_decided(monkeypatch):
    """The measurement this exists for, as a rule.

    "The solver did not decide" is a statement about an algorithm. Where
    the simplex stalled on these programmes the interior point method
    reached a verdict every time, so it is asked before the absence of a
    verdict is written down.
    """
    from faircanal import leximin as module

    fake = _fake_linprog({
        LP_METHOD: (4, "(HiGHS Status 0: Not Set)"),
        module.FALLBACK_METHOD: (2, "The problem is infeasible."),
    })
    monkeypatch.setattr(module, "linprog", fake)

    result, used, kind = module.run_linprog(
        np.zeros(4), sparse.csr_matrix((1, 4)), np.zeros(1), ((None, None),) * 4
    )
    assert fake.asked == [LP_METHOD, module.FALLBACK_METHOD]
    assert used == module.FALLBACK_METHOD
    assert kind is None
    assert result.status == 2


def test_the_second_algorithm_is_not_asked_when_the_first_one_answers(monkeypatch):
    """It can never change a number that already exists.

    That is the whole reason this is safe to add to a study whose results
    are already measured: the fallback fires only where the first
    algorithm produced no answer at all.
    """
    from faircanal import leximin as module

    for status, message in ((0, "Optimization terminated successfully."),
                            (2, "The problem is infeasible.")):
        fake = _fake_linprog({LP_METHOD: (status, message)})
        monkeypatch.setattr(module, "linprog", fake)
        _, used, kind = module.run_linprog(
            np.zeros(4), sparse.csr_matrix((1, 4)), np.zeros(1), ((None, None),) * 4
        )
        assert fake.asked == [LP_METHOD]
        assert used == LP_METHOD
        assert kind is None


def test_when_neither_algorithm_decides_the_answer_is_still_undecided(monkeypatch):
    """Two silences are not a verdict."""
    from faircanal import leximin as module

    fake = _fake_linprog({
        LP_METHOD: (4, "(HiGHS Status 0: Not Set)"),
        module.FALLBACK_METHOD: (1, "Time limit reached"),
    })
    monkeypatch.setattr(module, "linprog", fake)
    with pytest.raises(SolverFailure, match="neither algorithm decided"):
        solve_leximin(sharing_programme([1.0, 1.0], capacity=1.0))
    assert fake.asked == [LP_METHOD, module.FALLBACK_METHOD]


def test_the_fallback_is_not_asked_twice_when_it_is_the_one_configured(monkeypatch):
    from faircanal import leximin as module

    fake = _fake_linprog({module.FALLBACK_METHOD: (4, "(HiGHS Status 0: Not Set)")})
    monkeypatch.setattr(module, "linprog", fake)
    _, used, kind = module.run_linprog(
        np.zeros(4), sparse.csr_matrix((1, 4)), np.zeros(1), ((None, None),) * 4,
        method=module.FALLBACK_METHOD,
    )
    assert fake.asked == [module.FALLBACK_METHOD]
    assert kind is SolverFailure


def test_which_algorithm_answered_is_recorded_and_not_left_to_be_inferred(monkeypatch):
    """A result that needed the fallback says so, on the result itself."""
    from faircanal import leximin as module

    fake = _fake_linprog({
        LP_METHOD: (4, "(HiGHS Status 0: Not Set)"),
        module.FALLBACK_METHOD: (0, "Optimization terminated successfully."),
    })
    monkeypatch.setattr(module, "linprog", fake)
    solution = module._solve(
        np.zeros(4), sparse.csr_matrix((1, 4)), np.zeros(1), ((None, None),) * 4,
        None, None, "a programme that needed the second algorithm",
    )
    assert solution.solver_method == module.FALLBACK_METHOD


def test_the_two_algorithms_agree_where_both_decide():
    """On a real programme, not a stand-in.

    The fallback would be worthless if it answered a different question.
    """
    programme = sharing_programme([1.0, 2.0, 3.0], capacity=3.0)
    first = solve_leximin(programme)
    second = solve_leximin(programme, method="highs-ipm")
    assert np.allclose(first.ratios, second.ratios, atol=1.0e-7)
