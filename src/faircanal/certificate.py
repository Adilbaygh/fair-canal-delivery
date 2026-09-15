"""When an order cannot be met, say so - and say what is in the way.

The point of this module
------------------------
A method that always returns a schedule is a method that cannot tell a
farmer their order will not be filled. This one can. When the
lexicographic solution leaves anybody short, what goes back to the
allocation layer is not a plan but a statement: whose order, by how much,
what is blocking it, and what would have to change.

Two different failures, and they need different certificates
------------------------------------------------------------
The model document expects one of them: the programme has a solution but
somebody's order cannot be filled inside its window,

    min_i r_i < 1   =>   somebody's order cannot be met.

The test is on ``r_i``, never on the weighted ``r_i / w_i``: a user with a
weight below one has a weighted fraction above its real one, so a stage
level of one can still hide an unfilled order behind it.

The document also argues that this is the *only* failure, because ordering
nothing is always feasible. Under the framing this study settled on, that
argument does not hold and the difference matters. Orders are deviations
from the canal's nominal operation, so ordering nothing is every user
drawing its usual amount - not every user going without. When the source
is cut below what the users draw nominally, that point violates the source
constraint, and the deepest admissible reduction may violate the gates'
own limits before it reaches the source's. Then there is no schedule at
all, and the certificate has to say something else:

    no schedule exists   =>   what would have to give for one to exist.

Both are produced here. :func:`certify` takes the lexicographic result
when there is one and ``None`` when the solver could not find one, and
the relaxation it runs differs only in whether every fraction is held at
one or left free.

What the certificate carries
----------------------------
1. The fractions each user reached, the level at each lexicographic stage,
   and who saturated at each - so the answer to "who was cut, and when"
   is a name and a stage, not an inference.

2. The constraints that were active and what they were worth, from the
   dual of the last stage. This is the answer to "what is in the way":
   conveyance, the source, the level band, the gate's travel, the budget,
   or the delivered flow's own floor.

3. Each user's **structural ceiling** - the most it could get if nobody
   else were constrained at all. A ceiling below one means no allocation
   decision of any kind can fill that order, and the problem is in the
   infrastructure rather than in the sharing. It costs one programme per
   user and it is worth it, because it separates "you were outvoted" from
   "it was never possible".

4. A **multi-family elastic relaxation**: how much each family of
   constraints would have to give for everybody to be filled. One slack
   per family, not one overall, and emphatically not one family at a time
   - the audit that led to this module found a case where relaxing any
   single family alone left the problem unsolvable and only a combination
   worked. A single-family relaxation would have reported "impossible"
   about a situation that was merely expensive.

5. A digest of the inputs, so a certificate can be matched to the run that
   produced it.

The families
------------
They are the constraint groups the programme was assembled from, minus the
cap on the fraction - relaxing that would not relieve anything, it would
only redefine what being filled means. Each family's slack comes out in
its own unit, which is what makes the result readable: cubic metres per
second of source, metres of freeboard, cubic metres of budget.

Because those units differ, the objective divides each slack by a scale
taken from the scenario before adding them up, so that one unit of cost
means one whole scale of that family rather than one of whatever it
happens to be measured in. The scales are reported alongside the slacks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from faircanal.config import (
    CONVEYANCE_EFFICIENCY,
    solver_settings,
)
from faircanal.leximin import LeximinResult, run_linprog
from faircanal.programme import Programme

__all__ = [
    "CertificateError",
    "Family",
    "Binding",
    "Relaxation",
    "Certificate",
    "families",
    "structural_ceilings",
    "elastic_relaxation",
    "certify",
]

#: Families whose rows may be relaxed, and the unit each one's slack is in.
#:
#: Every row of the programme is here except the cap on the fraction,
#: which is not a limit on the canal at all: relaxing it would not fill an
#: order, it would only let over-delivery be counted as if it had.
#:
#: The nine divide into two kinds, and the report says which is which.
#: Seven of them - C2, C3, C5, C6, C7, C8, C9 - answer "what would the
#: canal have to give", and their slacks are metres and cubic metres a
#: canal could be built to. The other two answer a different question.
#: C5' says by how much the controller would have to be driven outside
#: the range its pool models were identified over, and C9' by how much
#: the volume account and the level account would have to be allowed to
#: disagree. Neither is something anybody can construct; both are
#: statements about the model rather than about the water.
#:
#: Leaving those two out was tried first and was worse. A programme whose
#: only obstacle is the identified range then reports that no relaxation
#: of the canal's limits produces a schedule, which is true and useless:
#: the reader is told the canal cannot do it when what stands in the way
#: is the model's own validity.
RELAXABLE = {
    "C2 delivered flow non-negative": "m^3/s",
    "C3 volume budget": "m^3",
    "C5 gate flow inside the reach's conveyance": "m^3/s",
    "C6 gate travel rate": "m^3/s per step",
    "C7 level inside its band": "m",
    "C8 source availability": "m^3/s",
    "C5' command inside the linear model's range": "m^3/s",
    "C9 pool storage between empty and full": "m^3",
    "C9' storage agrees with the level": "m^3",
}


class CertificateError(ValueError):
    """Raised when a certificate cannot be produced."""


@dataclass(frozen=True)
class Family:
    """One group of constraints, and the scale its slack is measured against."""

    name: str
    first: int
    last: int
    unit: str
    scale: float

    @property
    def rows(self) -> int:
        return self.last - self.first


@dataclass(frozen=True)
class Binding:
    """A family of constraints that was active, and what it was worth."""

    family: str
    active_rows: int
    largest_price: float
    unit: str


@dataclass(frozen=True)
class Relaxation:
    """How much each family would have to give for everybody to be filled."""

    feasible: bool
    slacks: dict
    scales: dict
    units: dict
    cost: float

    @property
    def binding_families(self) -> tuple[str, ...]:
        return tuple(
            name for name, value in sorted(self.slacks.items()) if value > 1e-9
        )


@dataclass(frozen=True)
class Certificate:
    """What goes back to the allocation layer when an order cannot be met."""

    scenario: str
    fulfilled: bool
    ratios: dict
    ceilings: dict
    stage_levels: tuple
    saturated: tuple
    binding: tuple
    relaxation: Relaxation
    digest: str
    shortfalls: dict = field(default_factory=dict)

    @property
    def impossible_users(self) -> tuple[str, ...]:
        """Users no allocation decision could have filled."""
        return tuple(
            name
            for name, ceiling in sorted(self.ceilings.items())
            if ceiling is not None and ceiling < 1.0 - 1e-9
        )

    def report(self) -> str:
        """The certificate in words, for the allocation layer and the log."""
        lines = [f"Delivery certificate for {self.scenario} [{self.digest[:12]}]"]
        if self.fulfilled:
            lines.append("Every order was filled inside its window.")
            return "\n".join(lines)

        if not self.ratios:
            lines.append(
                "NO SCHEDULE EXISTS: not one order can be met on these terms, "
                "however the water is shared."
            )
            if self.relaxation.feasible:
                given = ", ".join(
                    f"{name} +{self.relaxation.slacks[name]:.4g} "
                    f"{self.relaxation.units[name]}"
                    for name in self.relaxation.binding_families
                )
                if given:
                    lines.append(
                        f"A schedule would exist if these gave together: {given}."
                    )
                else:
                    # Every slack came back at zero, so the rows as written
                    # already admit a schedule and the caller asked the
                    # wrong question. Saying "these gave together: ." was
                    # worse than useless: an empty list read as an answer.
                    lines.append(
                        "No row had to give at all: a schedule exists under "
                        "these limits as they stand."
                    )
            else:
                lines.append(
                    "No relaxation of the canal's own limits produces a schedule; "
                    "the demands or the windows have to move."
                )
            return "\n".join(lines)

        worst = min(self.ratios, key=lambda name: self.ratios[name])
        lines.append(
            f"NOT FILLED: {len(self.shortfalls)} of {len(self.ratios)} orders. "
            f"Worst is {worst} at {self.ratios[worst]:.4f} of its demand "
            f"({self.shortfalls[worst]:.0f} m^3 short)."
        )
        impossible = self.impossible_users
        if impossible:
            lines.append(
                "No allocation decision could have filled: "
                + ", ".join(
                    f"{name} (ceiling {self.ceilings[name]:.4f})" for name in impossible
                )
                + " - the limit is in the canal, not in the sharing."
            )
        else:
            lines.append(
                "Every order is reachable on its own; what is missing is the "
                "water to fill them together."
            )
        if self.binding:
            lines.append("In the way, by what a unit of it is worth:")
            for entry in self.binding:
                lines.append(
                    f"  {entry.family}: {entry.active_rows} active, "
                    f"largest price {entry.largest_price:.6g} per {entry.unit}"
                )
        if self.relaxation.feasible:
            given = ", ".join(
                f"{name} +{self.relaxation.slacks[name]:.4g} "
                f"{self.relaxation.units[name]}"
                for name in self.relaxation.binding_families
            )
            lines.append(
                f"Everybody could be filled if these gave together: {given or 'nothing'}."
            )
        else:
            lines.append(
                "No relaxation of these families fills every order; the demands "
                "themselves have to move."
            )
        return "\n".join(lines)


def families(programme: Programme) -> tuple[Family, ...]:
    """Split the assembled rows into families, with a scale for each."""
    scenario = programme.scenario
    band = min(high for _, high in scenario.limits.level_band_m)
    headroom = min(
        capacity - nominal
        for capacity, nominal in zip(
            scenario.limits.capacity_m3_s, scenario.limits.nominal_m3_s
        )
    )
    area = np.array(programme.response.storage_area)
    depth = np.array([reach.pool.canal_depth_m for reach in scenario.network.reaches])
    target = np.array([reach.pool.target_level_m for reach in scenario.network.reaches])
    scales = {
        "C2 delivered flow non-negative": scenario.nominal_draw_m3_s,
        "C3 volume budget": scenario.aggregate_demand_m3,
        "C5 gate flow inside the reach's conveyance": headroom,
        "C6 gate travel rate": min(scenario.limits.travel_rate_m3_s),
        "C7 level inside its band": band,
        "C8 source availability": scenario.nominal_draw_m3_s,
        "C5' command inside the linear model's range": headroom,
        "C9 pool storage between empty and full": float(
            (area * (depth - target)).min()
        ),
        "C9' storage agrees with the level": float(
            (area * scenario.limits.band_tolerance_m).min()
        ),
    }
    out, cursor = [], 0
    for name, count in programme.row_counts.items():
        if name in RELAXABLE and count > 0:
            out.append(
                Family(
                    name=name,
                    first=cursor,
                    last=cursor + count,
                    unit=RELAXABLE[name],
                    scale=float(scales[name]),
                )
            )
        cursor += count
    if cursor != programme.ratio.a_ub.shape[0]:
        raise CertificateError(
            f"the row counts add to {cursor} but the programme has "
            f"{programme.ratio.a_ub.shape[0]} rows"
        )
    return tuple(out)


def _solve(cost, a_ub, b_ub, bounds, what):
    solution, used, kind = run_linprog(cost, a_ub, b_ub, bounds)
    if kind is not None:
        # Not a verdict. A certificate is a statement about the canal; one
        # built on a solver that did not decide is a statement about
        # nothing - see faircanal.leximin.SolverUndecided.
        error = kind(
            f"{what}: neither algorithm decided; {used} returned status "
            f"{solution.status}: {solution.message}"
        )
        error.status = int(solution.status)
        error.method = used
        raise error
    if not solution.success:
        raise CertificateError(f"{what}: {solution.message}")
    return solution


def structural_ceilings(programme: Programme) -> dict:
    """The most each user could get with nobody else in the way.

    One programme per user: maximise that user's own fraction over the
    whole feasible set, with no floor on anybody else. A ceiling below one
    says the order is unreachable however the water is shared, which is a
    different message from "you were outvoted" and has to be sent as one.
    """
    ratio = programme.ratio
    rows = ratio.ratio_rows.toarray()
    ceilings = {}
    for index, name in enumerate(ratio.names):
        solution = _solve(
            -rows[index],
            ratio.a_ub,
            ratio.b_ub,
            ratio.bounds,
            f"structural ceiling for {name}",
        )
        ceilings[name] = float(-solution.fun + ratio.ratio_offset[index])
    return ceilings


def elastic_relaxation(
    programme: Programme,
    weights: dict | None = None,
    fulfil: bool = True,
    allowed: "set[str] | None" = None,
) -> Relaxation:
    """What every family would have to give for every order to be filled.

    One slack per family, all free at once. Relaxing them one at a time is
    the mistake this replaced: a real scenario in the audit needed the
    source *and* the conveyance to give together, and neither alone was
    enough, so a one-at-a-time search reported an impossibility that was
    not one.

    With ``fulfil`` false the fractions are left free and the question
    becomes the weaker one - what would have to give for any schedule at
    all to exist. That is the form the certificate takes when the
    lexicographic solver could not start.

    ``allowed`` pins every family outside it shut. It exists so that the
    one-at-a-time search can be run on purpose and shown to give a
    different, worse answer than letting the families give together.
    """
    ratio = programme.ratio
    groups = families(programme)
    if not groups:
        raise CertificateError("no relaxable family in this programme")

    n_vars, n_extra = ratio.n_vars, len(groups)
    selector = np.zeros((ratio.a_ub.shape[0], n_extra))
    for column, group in enumerate(groups):
        selector[group.first : group.last, column] = 1.0

    a_ub = sparse.hstack(
        [ratio.a_ub, sparse.csr_matrix(-selector)], format="csr"
    )
    b_ub = ratio.b_ub

    if fulfil:
        # every fraction at least one, on the unweighted r
        floors = sparse.hstack(
            [-ratio.ratio_rows, sparse.csr_matrix((ratio.n_users, n_extra))],
            format="csr",
        )
        a_ub = sparse.vstack([a_ub, floors], format="csr")
        b_ub = np.concatenate([b_ub, ratio.ratio_offset - 1.0])

    if weights is None:
        weights = {group.name: 1.0 / group.scale for group in groups}
    cost = np.concatenate(
        [np.zeros(n_vars), np.array([weights[group.name] for group in groups])]
    )
    bounds = (
        *ratio.bounds,
        *(
            (0.0, None) if (allowed is None or group.name in allowed) else (0.0, 0.0)
            for group in groups
        ),
    )

    # Through run_linprog like every other programme in this package, and
    # not through linprog directly. The difference is not style: a solver
    # that stopped without deciding used to be written down here as
    # "no relaxation of the canal's own limits produces a schedule", which
    # is a statement about the canal made from a statement about the
    # solver. A verdict that does not exist is raised, not reported.
    try:
        solution = _solve(cost, a_ub, b_ub, bounds, "the elastic relaxation")
    except CertificateError:
        return Relaxation(
            feasible=False,
            slacks={group.name: float("inf") for group in groups},
            scales={group.name: group.scale for group in groups},
            units={group.name: group.unit for group in groups},
            cost=float("inf"),
        )
    return Relaxation(
        feasible=True,
        slacks={
            group.name: float(solution.x[n_vars + index])
            for index, group in enumerate(groups)
        },
        scales={group.name: group.scale for group in groups},
        units={group.name: group.unit for group in groups},
        cost=float(solution.fun),
    )


#: Significant digits every number in the digest payload is rounded to
#: before hashing. See :func:`_canonical`.
DIGEST_DIGITS = 10


def _canonical(value):
    """The digest payload with every number rounded to a declared precision.

    A digest is supposed to answer one question: were these two runs the
    same problem under the same settings? Several of the numbers in the
    payload are not declared but computed - a reach's conveyance comes out
    of a Manning solve, a pool's backwater area out of the identified
    model - and two machines with different BLAS builds can produce those
    to the last bit differently while agreeing on every digit anyone
    reports. Hashed raw, the two runs then get different digests, and the
    digest stops meaning "a different problem" and starts meaning "a
    different machine", which is not a question anybody asked it.

    Rounding to :data:`DIGEST_DIGITS` significant digits fixes that, and
    the cost has to be stated rather than hidden: a change in how a
    coefficient is derived that moves it by less than one part in
    :math:`10^{10}` will no longer show up in the digest. Nothing in this
    study derives a coefficient to that precision - the conveyances are
    reported to four decimals and the solver's own feasibility tolerance
    is :math:`10^{-9}` - so what is given up is noise, and what is bought
    is a digest that two machines can compare.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        if value == 0.0:
            return 0.0
        exponent = math.floor(math.log10(abs(value)))
        return round(value, DIGEST_DIGITS - 1 - exponent)
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _digest(programme: Programme) -> str:
    """A hash of everything the answer depends on."""
    text = json.dumps(
        _canonical(digest_payload(programme)), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_payload(programme: Programme) -> dict:
    """What the digest is taken over, as a dictionary.

    Separated from the hashing so that a mismatch between two machines
    can be located block by block rather than guessed at;
    ``scripts/check_digest.py`` does exactly that.
    """
    scenario = programme.scenario
    payload = {
        "scenario": scenario.name,
        "network": scenario.network.name,
        "topology": scenario.network.topology_rule,
        "blocks": scenario.blocks,
        "steps_per_block": scenario.steps_per_block,
        "dt_s": scenario.dt_s,
        "horizon": scenario.horizon,
        "source_discharge_m3_s": scenario.source_discharge_m3_s,
        "source_nominal_until": scenario.source_nominal_until,
        "source_ramp_steps": scenario.source_ramp_steps,
        "ratio_cap": scenario.ratio_cap,
        "demand_floor_m3": scenario.demand_floor_m3,
        "loss_factor": scenario.loss_factor,
        "warm_up_steps": scenario.limits.warm_up_steps,
        "capacity_m3_s": list(scenario.limits.capacity_m3_s),
        # The gate limit, and not only the reach's conveyance. Leaving it
        # out was caught the way the filter's absence was caught - by two
        # runs that gave different answers sharing a digest. At half
        # supply the scan and the narrow-gate sensitivity run returned
        # 0.6915 and 0.6534 under the same sixteen hex digits, because the
        # payload described the reach and not the gate the water has to
        # pass to get into it. A gate with no published limit is null, not
        # a large number: "none published" and "this wide" are different
        # claims and must hash differently.
        "gate_capacity_m3_s": (
            None
            if scenario.limits.gate_capacity_m3_s is None
            else [
                None if math.isinf(value) else value
                for value in scenario.limits.gate_capacity_m3_s
            ]
        ),
        "nominal_m3_s": list(scenario.limits.nominal_m3_s),
        "level_band_m": [list(band) for band in scenario.limits.level_band_m],
        "travel_rate_m3_s": list(scenario.limits.travel_rate_m3_s),
        # What the answer depends on, not what happens to be nearby. The
        # first version of this block read the filter's order off
        # ``delivery.shape[1]``, which is the number of order blocks, and
        # carried neither the cut-off nor the control law nor the solver.
        # Two runs that differed only in the filter therefore produced the
        # same digest, and the collision was reported in the article with
        # the wrong cause attached to it.
        "filter": {
            "order": programme.response.filter_order,
            "cutoff_rad_per_s": programme.response.filter_cutoff_rad_per_s,
            "sample_time_s": scenario.dt_s,
            "steps": programme.steps,
        },
        "control": programme.response.control_signature,
        "storage": {
            "area": [float(value) for value in programme.response.storage_area],
            "transport_lag": list(programme.response.transport_lag),
            "band_tolerance_m": scenario.limits.band_tolerance_m,
            "conveyance_efficiency": CONVEYANCE_EFFICIENCY,
        },
        "solver": solver_settings(),
        "users": [
            {
                "name": user.name,
                "node": user.node,
                "nominal_m3_s": user.nominal_m3_s,
                "demand_m3": user.demand_m3,
                "window": list(user.window),
                "announced_block": user.announced_block,
                "max_order_m3_s": user.max_order_m3_s,
                "weight": user.weight,
                "overshoot": user.overshoot,
            }
            for user in scenario.users
        ],
    }
    return payload


def certify_no_schedule(programme: Programme) -> Certificate:
    """The certificate for a programme that has no solution at all.

    There are no fractions to report and no duals to read, so what is left
    is the question that still has an answer: what would have to give for
    any schedule to exist. Every user's structural ceiling is computed too,
    because a ceiling is defined by the feasible set of one user's own
    programme and can exist even when the joint one does not.
    """
    relaxation = elastic_relaxation(programme, fulfil=False)
    try:
        ceilings = structural_ceilings(programme)
    except CertificateError:
        # Not a number that happens to be missing - a quantity that is not
        # defined here, because a ceiling is the optimum of a programme that
        # has none. It is written as null rather than NaN: NaN is not valid
        # JSON, and a results file a strict reader refuses is not archivable.
        ceilings = {name: None for name in programme.ratio.names}
    return Certificate(
        scenario=programme.scenario.name,
        fulfilled=False,
        ratios={},
        ceilings=ceilings,
        stage_levels=(),
        saturated=(),
        binding=(),
        relaxation=relaxation,
        digest=_digest(programme),
        shortfalls={
            user.name: float(user.demand_m3) for user in programme.scenario.users
        },
    )


def certify(
    programme: Programme,
    result: LeximinResult | None,
    with_ceilings: bool = True,
    with_relaxation: bool = True,
) -> Certificate:
    """Turn a solved programme into the object the allocation layer reads.

    ``with_ceilings`` and ``with_relaxation`` are there because each costs
    programmes of its own - one per user, and one more - and a scan over
    many supply levels does not need them at every point. They default to
    on, because a certificate without them answers "you did not get it"
    and not "here is why".
    """
    if result is None:
        return certify_no_schedule(programme)
    weights = programme.ratio.weights
    unweighted = result.ratios * weights
    ratios = {
        name: float(value) for name, value in zip(programme.ratio.names, unweighted)
    }
    cap = programme.scenario.ratio_cap
    shortfalls = {}
    for user in programme.scenario.users:
        missing = (cap - ratios[user.name]) * user.demand_m3
        if missing > 1e-6:
            shortfalls[user.name] = float(missing)

    binding = []
    marginals = np.asarray(result.marginals, dtype=float)
    rows = programme.ratio.a_ub.shape[0]
    if marginals.size >= rows:
        # The staged programme stacks the model's own rows first and its
        # floors after, so the model's duals are the leading entries.
        marginals = marginals[:rows]
        for group in families(programme):
            prices = np.abs(marginals[group.first : group.last])
            active = int((prices > 1e-9).sum())
            if active:
                binding.append(
                    Binding(
                        family=group.name,
                        active_rows=active,
                        largest_price=float(prices.max()),
                        unit=group.unit,
                    )
                )

    fulfilled = not shortfalls
    ceilings = (
        structural_ceilings(programme)
        if (with_ceilings and not fulfilled)
        else {name: cap for name in programme.ratio.names}
    )
    relaxation = (
        elastic_relaxation(programme)
        if (with_relaxation and not fulfilled)
        else Relaxation(feasible=True, slacks={}, scales={}, units={}, cost=0.0)
    )

    return Certificate(
        scenario=programme.scenario.name,
        fulfilled=fulfilled,
        ratios=ratios,
        ceilings=ceilings,
        stage_levels=tuple(result.stage_levels),
        saturated=tuple(result.stage_saturated),
        binding=tuple(binding),
        relaxation=relaxation,
        digest=_digest(programme),
        shortfalls=shortfalls,
    )
