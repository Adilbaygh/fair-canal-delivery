"""Permanent tests for the delivery certificate. Never deleted.

The certificate is the part of the method that says "no". Everything else
in this package produces a number; this produces a refusal with reasons,
and a refusal that names the wrong reason is worse than no refusal at all.
One test here exists for a mistake the model document's own audit caught:
relaxing the constraints one family at a time, which reports an
impossibility about a situation that is merely expensive.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from faircanal.benchmarks import haughton_filter
from faircanal.certificate import (
    RELAXABLE,
    CertificateError,
    _canonical,
    _digest,
    certify,
    digest_payload,
    elastic_relaxation,
    families,
    structural_ceilings,
)
from faircanal.control import control_law
from faircanal.geometry import uniform_discharge
from faircanal.leximin import LeximinError, solve_leximin
from faircanal.network import corning_cascade
from faircanal.plant import build_plant, horizon_for, response_map
from faircanal.programme import assemble
from faircanal.scenario import Limits, Scenario, one_user_per_gate

SPEC = haughton_filter(3)
BLOCKS = 6
LEAD = 2


@lru_cache(maxsize=2)
def plant():
    return build_plant(corning_cascade(), SPEC)


@lru_cache(maxsize=2)
def mapping():
    steps = horizon_for(BLOCKS)
    return response_map(
        plant(),
        BLOCKS,
        law=control_law(list(plant().design_models), plant().weights, horizon=steps),
    )


def make_limits(capacity_scale: float = 1.0, warm: int = 15) -> Limits:
    net = corning_cascade()
    full = [
        uniform_discharge(reach.pool, reach.pool.canal_depth_m)
        for reach in net.reaches
    ]
    nominal = net.steady_discharges
    return Limits(
        capacity_m3_s=tuple(
            base + (top - base) * capacity_scale for top, base in zip(full, nominal)
        ),
        nominal_m3_s=nominal,
        level_band_m=tuple(
            (
                -(reach.pool.canal_depth_m - reach.pool.target_level_m),
                reach.pool.canal_depth_m - reach.pool.target_level_m,
            )
            for reach in net.reaches
        ),
        travel_rate_m3_s=tuple(0.25 * value for value in full),
        warm_up_steps=warm,
    )


@lru_cache(maxsize=8)
def model(fraction: float, capacity_scale: float = 1.0, demand_scale: float = 1.0):
    net = corning_cascade()
    scenario = one_user_per_gate(
        net,
        BLOCKS,
        make_limits(capacity_scale),
        source_discharge_m3_s=fraction * net.aggregate_demand,
        lead_blocks=LEAD,
        demand_scale=demand_scale,
    )
    return assemble(scenario, mapping())


@lru_cache(maxsize=8)
def certificate(fraction: float, capacity_scale: float = 1.0):
    programme = model(fraction, capacity_scale)
    try:
        result = solve_leximin(programme.ratio)
    except LeximinError:
        result = None
    return certify(programme, result)


# ---------------------------------------------------------------------------
# What it says
# ---------------------------------------------------------------------------


def test_a_full_supply_certificate_says_everything_was_filled():
    certified = certificate(1.0)
    assert certified.fulfilled
    assert not certified.shortfalls
    assert "filled" in certified.report()
    assert all(value == pytest.approx(1.0) for value in certified.ratios.values())


def test_a_short_certificate_names_the_worst_off_and_by_how_much():
    """Whose order, by how much, and in what units - not a status code."""
    certified = certificate(0.7)
    assert not certified.fulfilled
    assert certified.shortfalls
    worst = min(certified.ratios, key=lambda name: certified.ratios[name])
    assert worst in certified.shortfalls
    assert certified.shortfalls[worst] > 0.0
    report = certified.report()
    assert "NOT FILLED" in report
    assert worst in report
    assert "m^3" in report


def test_the_ceiling_separates_being_outvoted_from_being_impossible():
    """Two different messages, and the certificate has to tell them apart.

    A user can be short because the water went elsewhere, or because no
    allocation decision of any kind could have filled it. The first is a
    sharing outcome; the second is a statement about the canal. At mild
    scarcity every order is still reachable on its own; deeper down some
    are not, and the certificate names them.
    """
    mild = certificate(0.7)
    assert not mild.fulfilled
    assert mild.impossible_users == ()
    assert "reachable on its own" in mild.report()

    # Fifty-five per cent, not fifty: with the gate limit and the storage
    # state in the programme this six-block instance runs out of schedules
    # one step of the scan earlier than it used to.
    deep = certificate(0.55)
    assert deep.impossible_users, "no user is now structurally short"
    assert "No allocation decision could have filled" in deep.report()
    for name in deep.impossible_users:
        assert deep.ceilings[name] < 1.0


def test_the_ceiling_is_the_most_one_user_could_get_alone():
    """Every ceiling is above what that user actually got, and at most one."""
    programme = model(0.55)
    result = solve_leximin(programme.ratio)
    certified = certify(programme, result)
    ceilings = structural_ceilings(programme)
    for name, ceiling in ceilings.items():
        assert ceiling <= programme.scenario.ratio_cap + 1e-9
        assert ceiling >= certified.ratios[name] - 1e-6, (
            f"{name} got more together than it could get alone"
        )


# ---------------------------------------------------------------------------
# The mistake the audit caught
# ---------------------------------------------------------------------------


def test_one_family_at_a_time_reports_an_impossibility_that_is_not_one():
    """Why the relaxation frees every family at once.

    Pinned to one family, the canal cannot be made to fill every order at
    all - every one of the nine alone is infeasible. Freed together they
    can, and the answer names three of them. A search that tried them one
    at a time would have reported that nothing could be done, about a
    situation that only needed several things to give a little each.

    The scenario is the one the adequacy audit found the mistake in,
    re-tuned for the programme as it now stands: the conveyance squeezed
    to a tenth and the demand raised by fifteen per cent. The audit's own
    numbers - a fiftieth and thirty per cent - are past the point where
    any relaxation helps, because C1 now caps a user's outlet and no
    amount of canal fills an order the outlet cannot pass. That refusal is
    the certificate working, and it is checked below.
    """
    programme = model(0.9, 0.1, 1.15)
    joint = elastic_relaxation(programme)
    assert joint.feasible
    assert len(joint.binding_families) >= 3, joint.binding_families

    for group in families(programme):
        alone = elastic_relaxation(programme, allowed={group.name})
        assert not alone.feasible, (
            f"{group.name} alone is now enough, so this test no longer shows "
            f"why the families are freed together"
        )


def test_an_order_the_outlet_cannot_pass_is_refused_by_no_amount_of_canal():
    """C1 is the one limit the canal cannot be relaxed around.

    Every other family is something that could be built bigger. A user's
    own outlet is not part of the programme's rows at all - it is a bound
    on the decision - so when the demand exceeds what that outlet can pass
    inside the window, freeing every row at once still fills nothing. The
    certificate says so rather than returning a schedule that does not
    exist.
    """
    programme = model(0.9, 0.02, 1.3)
    joint = elastic_relaxation(programme)
    assert not joint.feasible
    for group in families(programme):
        assert not elastic_relaxation(programme, allowed={group.name}).feasible
    # And a schedule does exist here - it simply cannot fill every order,
    # which is the distinction the two relaxations are there to keep.
    assert elastic_relaxation(programme, fulfil=False).feasible


def test_the_joint_relaxation_is_never_worse_than_a_single_family():
    programme = model(0.7)
    joint = elastic_relaxation(programme)
    assert joint.feasible
    for group in families(programme):
        alone = elastic_relaxation(programme, allowed={group.name})
        if alone.feasible:
            assert joint.cost <= alone.cost + 1e-9


def test_the_slacks_come_out_in_their_own_units():
    programme = model(0.7)
    relaxation = elastic_relaxation(programme)
    assert set(relaxation.units) == set(relaxation.slacks)
    assert relaxation.units["C8 source availability"] == "m^3/s"
    assert relaxation.units["C7 level inside its band"] == "m"
    assert relaxation.units["C3 volume budget"] == "m^3"
    assert all(scale > 0.0 for scale in relaxation.scales.values())
    # The source is what is short here, and by a sensible amount.
    short = relaxation.slacks["C8 source availability"]
    assert 0.0 < short < programme.scenario.nominal_draw_m3_s


# ---------------------------------------------------------------------------
# When there is no schedule at all
# ---------------------------------------------------------------------------


def test_a_programme_with_no_schedule_still_says_what_would_make_one():
    """The failure the model document did not expect, and why it happens.

    The document argues the programme can never be infeasible because
    ordering nothing is always feasible. Under this study's framing that is
    not so: orders are deviations, so ordering nothing is every user
    drawing its usual amount, and when the source is cut below that the
    point is not feasible at all. The certificate then answers the weaker
    question instead of returning nothing.
    """
    programme = model(0.3)
    with pytest.raises(LeximinError):
        solve_leximin(programme.ratio)

    certified = certify(programme, None)
    assert not certified.fulfilled
    assert certified.ratios == {}
    # A ceiling is the optimum of a programme that has none here, so it is
    # not a number - and it is written as one that JSON has. NaN would be
    # read back by Python and refused by everything else.
    assert set(certified.ceilings.values()) == {None}
    assert certified.impossible_users == ()
    report = certified.report()
    assert "NO SCHEDULE EXISTS" in report
    assert certified.relaxation.feasible
    assert certified.relaxation.binding_families
    assert "would exist if" in report


# ---------------------------------------------------------------------------
# The reading that has to be right
# ---------------------------------------------------------------------------


def test_fulfilment_is_read_from_the_unweighted_fraction():
    """The correction the model document makes to itself, kept in force.

    A priority weight below one lifts the weighted fraction above the real
    one, so a lexicographic stage can sit at one while the order behind it
    is only part filled. What decides whether an order was met is always
    the plain ratio of volume delivered to volume asked for.
    """
    base = model(0.7)
    weighted_users = tuple(
        replace(user, weight=0.5) if index == 0 else user
        for index, user in enumerate(base.scenario.users)
    )
    scenario = Scenario(
        name="weighted",
        network=base.scenario.network,
        users=weighted_users,
        blocks=base.scenario.blocks,
        limits=base.scenario.limits,
        source_discharge_m3_s=base.scenario.source_discharge_m3_s,
        source_nominal_until=base.scenario.source_nominal_until,
    )
    programme = assemble(scenario, mapping())
    result = solve_leximin(programme.ratio)
    certified = certify(programme, result, with_ceilings=False, with_relaxation=False)

    name = weighted_users[0].name
    assert certified.ratios[name] == pytest.approx(result.ratios[0] * 0.5, rel=1e-9)
    assert result.ratios[0] > certified.ratios[name], (
        "the weighted fraction is no longer above the real one, so this test "
        "no longer distinguishes them"
    )
    assert certified.ratios[name] <= programme.scenario.ratio_cap + 1e-9
    if certified.ratios[name] < 1.0 - 1e-9:
        assert name in certified.shortfalls


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


def test_the_families_cover_every_row_except_the_cap():
    programme = model(0.7)
    groups = families(programme)
    assert {group.name for group in groups} == set(RELAXABLE)
    covered = sum(group.rows for group in groups)
    assert covered == programme.ratio.a_ub.shape[0] - programme.scenario.n_users
    for earlier, later in zip(groups, groups[1:]):
        assert earlier.last <= later.first


def test_the_binding_families_are_reported_with_their_prices():
    certified = certificate(0.55)
    assert certified.binding
    names = {entry.family for entry in certified.binding}
    assert "C8 source availability" in names
    for entry in certified.binding:
        assert entry.active_rows > 0
        assert entry.largest_price > 0.0
        assert entry.unit == RELAXABLE[entry.family]


def test_the_digest_follows_the_inputs():
    """Same question, same digest; different question, different digest."""
    first = certificate(0.7).digest
    assert certificate(0.7).digest == first
    assert certificate(0.5).digest != first
    assert len(first) == 64


def test_the_digest_survives_a_different_machine_and_not_a_different_problem():
    """Rounded, and rounded on purpose.

    Two machines with different BLAS builds compute a reach's conveyance
    - a Manning solve - and a pool's backwater area to the last bit
    differently while agreeing on every digit anybody reports. Hashed
    raw, the same instance then gets two digests, and the digest answers
    "a different machine" instead of "a different problem".

    So the payload is rounded to ``DIGEST_DIGITS`` significant figures
    first. What that costs is checked here as well as what it buys: a
    change of one part in a million still changes the digest, which is
    far finer than anything this study derives or reports.
    """
    programme = model(0.7)
    payload = digest_payload(programme)
    before = _digest(programme)

    def hashed(block):
        text = json.dumps(_canonical(block), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    baseline = hashed(payload)
    assert len(baseline) == 64

    # Last-bit noise: invisible, which is the point.
    nudged = deepcopy(payload)
    nudged["capacity_m3_s"][0] *= 1.0 + 1.0e-14
    nudged["storage"]["area"][0] *= 1.0 - 1.0e-14
    assert hashed(nudged) == baseline

    # A real change: visible, which is also the point.
    changed = deepcopy(payload)
    changed["capacity_m3_s"][0] *= 1.0 + 1.0e-6
    assert hashed(changed) != baseline

    # And the rounding did not quietly turn every digest into the same one.
    assert _digest(programme) == before
    assert _digest(model(0.5)) != before


def test_the_digest_separates_two_filters_that_differ():
    """The collision this digest was fixed to stop.

    The payload used to name the scenario and not the loop, so a run with
    a different filter produced the same digest as the main scan - and
    the article reported that collision with the wrong cause attached to
    it. Rounding must not bring it back: two filters are two problems,
    and the scenario alone cannot tell them apart.
    """
    fourth_order = build_plant(corning_cascade(), haughton_filter(4))
    steps = horizon_for(BLOCKS)
    other = response_map(
        fourth_order,
        BLOCKS,
        law=control_law(
            list(fourth_order.design_models), fourth_order.weights, horizon=steps
        ),
    )
    scenario = model(0.7).scenario
    assert _digest(assemble(scenario, other)) != _digest(model(0.7))


def test_a_programme_with_nothing_to_relax_is_refused():
    programme = model(0.7)
    empty = replace(programme, row_counts={"ratio capped at one": programme.ratio.a_ub.shape[0]})
    with pytest.raises(CertificateError, match="no relaxable family"):
        elastic_relaxation(empty)


def test_row_counts_that_do_not_add_up_are_refused():
    programme = model(0.7)
    broken = replace(programme, row_counts={"C8 source availability": 3})
    with pytest.raises(CertificateError, match="row counts"):
        families(broken)
