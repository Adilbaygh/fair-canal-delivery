"""Who orders what, when, and what the canal is allowed to do about it.

Everything in this module is a scenario parameter: nothing here is
published for this canal, so every field is ``assumed`` and every value has
to be written down with the result it produced. The module's job is to
make that impossible to skip - a scenario cannot be built without stating
each of them, and it refuses the combinations that would make the
fairness measure meaningless.

The one thing that is not a free choice
---------------------------------------
Orders are **deviations from the canal's nominal operation**. The pool
models were identified at the published steady state, in which every gate
already draws its published offtake, so that state is where the linear
model has its zero. A user who orders nothing is a user drawing their
usual amount, not a user with a closed gate.

Everything a user experiences is still absolute: the demand ``D_i`` is the
volume asked for, the delivered volume counts the nominal draw as well as
the deviation, and the ratio ``r_i`` compares the two. Only the decision
variable is a deviation, and only because that is where the plant's zero
is. Getting this backwards puts a step change of the whole nominal offtake
through the filter at the first sample, which is a transient no canal ever
experiences.

When the restriction starts
---------------------------
A canal put on restriction does not become short at the first sample. The
users are drawing their usual amount when the run begins, and the filter
that stands between an order and the water is slow: at the first step its
impulse response has barely begun, so no order however small can reduce
the flow yet. Writing the source constraint as though the shortage were
already in force at step zero makes the programme infeasible for every
schedule, which is not a finding about water - it is a statement that the
run started in the middle of something.

So the restriction has a start. ``source_nominal_until`` is the step from
which the source discharge applies; before it the canal releases what the
users draw nominally. It is ``assumed``, it is one integer, and it means
something a canal operator would recognise: when the notice takes effect.

It also has a rate. A restriction that steps from full supply to short
supply between two samples asks the users to change their draw faster than
the filter between an order and the water allows, and the programme is
infeasible at exactly the two steps where the profile jumps - by five per
cent of the discharge, at two steps out of two hundred, whatever the
schedule. Nothing about that is a finding. So the profile ramps over
``source_ramp_steps``, which is one order block by default: the notice
takes effect at the rate the orders themselves are allowed to change.

It also has an end, and that one is not a choice. Orders exist only for
the blocks, and the run continues past the last of them so that the
filter's tail lands inside it. Over that tail the deviation decays to
nothing whatever anybody ordered, so a restriction still in force there
asks for a reduction no schedule can produce, and the programme is
infeasible for reasons that have nothing to do with water. The restriction
therefore ends with the last block, and normal service resumes while the
last order is still arriving.

Why a demand can be refused
---------------------------
``r_i = V_i / D_i`` has ``D_i`` in the denominator. A user whose demand is
nearly zero reaches a huge ratio on a spoonful of water and walks off with
the lexicographic order; a user whose demand is exactly zero has no ratio
at all. So a floor is declared, users below it are excluded, and the
excluded list is part of the result rather than a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faircanal.config import (
    ANNOUNCE_BLOCKS,
    BAND_TOLERANCE_M,
    D_MIN_M3,
    DT_PLANT_S,
    OUTLET_HEADROOM,
    R_CAP,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.network import Network

__all__ = [
    "ScenarioError",
    "User",
    "Limits",
    "Scenario",
    "one_user_per_gate",
]


class ScenarioError(ValueError):
    """Raised when a scenario would make the fairness measure meaningless."""


@dataclass(frozen=True)
class User:
    """One offtake, with what it asks for and when it counts.

    ``nominal_m3_s`` is this user's share of the published offtake at its
    gate - the flow it draws when it orders nothing. ``demand_m3`` is the
    volume it asks for inside its window, counted from zero and not from
    the nominal, so a user asking for exactly its usual amount has a demand
    equal to ``nominal_m3_s`` times the window's length.

    ``window`` is inclusive at both ends and is in plant steps.
    ``announced_block`` is the first block whose order the user may change:
    before it the deviation is held at zero, which is causality, not a
    preference.
    """

    name: str
    node: int
    nominal_m3_s: float
    demand_m3: float
    window: tuple[int, int]
    announced_block: int = 0
    max_order_m3_s: float | None = None
    weight: float = 1.0
    overshoot: float = 0.0

    def __post_init__(self) -> None:
        if self.node < 1:
            raise ScenarioError(f"{self.name}: node numbers start at one")
        if self.nominal_m3_s < 0.0:
            raise ScenarioError(f"{self.name}: a nominal draw cannot be negative")
        if self.demand_m3 <= 0.0:
            raise ScenarioError(
                f"{self.name}: a demand of {self.demand_m3} leaves the ratio "
                f"undefined; exclude the user instead of giving it nothing to ask "
                f"for"
            )
        first, last = self.window
        if not 0 <= first <= last:
            raise ScenarioError(f"{self.name}: the window {self.window} is empty")
        if self.announced_block < 0:
            raise ScenarioError(f"{self.name}: an announcement block cannot precede zero")
        if self.weight <= 0.0:
            raise ScenarioError(
                f"{self.name}: a weight of {self.weight} would make the fraction "
                f"undefined rather than the user unimportant"
            )
        if self.overshoot < 0.0:
            raise ScenarioError(f"{self.name}: the delivery allowance cannot be negative")
        if self.max_order_m3_s is not None and self.max_order_m3_s < self.nominal_m3_s:
            raise ScenarioError(
                f"{self.name}: its own outlet is capped at {self.max_order_m3_s} m^3/s, "
                f"below the {self.nominal_m3_s} m^3/s it already draws"
            )

    @property
    def window_steps(self) -> int:
        first, last = self.window
        return last - first + 1

    def nominal_volume_m3(self, dt: float = DT_PLANT_S) -> float:
        """What the user receives inside its window if it orders nothing."""
        return self.nominal_m3_s * self.window_steps * dt


@dataclass(frozen=True)
class Limits:
    """What the canal is allowed to do, per reach.

    None of these is published for this canal. ``capacity_m3_s`` follows
    from the geometry - what the reach conveys at full supply level - and
    is therefore ``derived``; the level band, the gate travel rate and the
    warm-up are ``assumed`` and have to be reported with any result they
    produced.

    ``band_tolerance_m`` is the width C9' reconciles the two accounts of
    the same water to: the volume that went in and out of a pool, and the
    volume its mean level says it holds. It is a *measured* quantity, not
    a chosen one - ``scripts/check_storage_band.py`` runs the criterion
    and reads the discrepancy off the schedule the criterion itself picks
    - and it lives here rather than in :mod:`faircanal.config` because it
    describes this canal under this model, so a sensitivity run that
    widens or narrows it is describing a different canal and the digest
    has to say so.
    """

    capacity_m3_s: tuple[float, ...]
    nominal_m3_s: tuple[float, ...]
    level_band_m: tuple[tuple[float, float], ...]
    travel_rate_m3_s: tuple[float, ...]
    gate_capacity_m3_s: "tuple[float, ...] | None" = None
    band_tolerance_m: float = BAND_TOLERANCE_M
    warm_up_steps: int = 0
    provenance: str = "derived capacity and gate limit, assumed band and travel rate"

    def __post_init__(self) -> None:
        size = len(self.capacity_m3_s)
        for name, values in (
            ("nominal_m3_s", self.nominal_m3_s),
            ("level_band_m", self.level_band_m),
            ("travel_rate_m3_s", self.travel_rate_m3_s),
        ):
            if len(values) != size:
                raise ScenarioError(
                    f"{name} has {len(values)} entries for {size} reaches"
                )
        for index, (capacity, nominal) in enumerate(
            zip(self.capacity_m3_s, self.nominal_m3_s), start=1
        ):
            if capacity <= nominal:
                raise ScenarioError(
                    f"reach {index}: its conveyance of {capacity:.3f} m^3/s is not "
                    f"above the {nominal:.3f} m^3/s it already carries, so there is "
                    f"no room to deliver anything"
                )
        for index, (low, high) in enumerate(self.level_band_m, start=1):
            if not low < 0.0 < high:
                raise ScenarioError(
                    f"reach {index}: the level band {(low, high)} does not contain "
                    f"the set-point"
                )
        if any(rate <= 0.0 for rate in self.travel_rate_m3_s):
            raise ScenarioError("a gate that cannot move has no travel rate")
        if self.gate_capacity_m3_s is not None:
            if len(self.gate_capacity_m3_s) != size:
                raise ScenarioError(
                    f"gate_capacity_m3_s has {len(self.gate_capacity_m3_s)} "
                    f"entries for {size} reaches"
                )
            for index, (gate, nominal) in enumerate(
                zip(self.gate_capacity_m3_s, self.nominal_m3_s), start=1
            ):
                if gate <= nominal:
                    raise ScenarioError(
                        f"reach {index}: its gate passes {gate:.3f} m^3/s at the "
                        f"declared head, which is not above the {nominal:.3f} "
                        f"m^3/s it already carries, so no order can be filled"
                    )
        if self.band_tolerance_m <= 0.0:
            raise ScenarioError(
                "the two accounts of the same water are reconciled to within a "
                "band, and a band of zero or less says the pool's mean level "
                "and its volume agree exactly, which no lumped model of a "
                "canal does"
            )
        if self.warm_up_steps < 0:
            raise ScenarioError("a warm-up cannot be negative")

    @property
    def size(self) -> int:
        return len(self.capacity_m3_s)

    @property
    def flow_bounds(self) -> tuple[tuple[float, float], ...]:
        """C5: what the gate may actually pass, as a flow deviation.

        Two physical limits on the same variable, so one bound: the reach
        cannot carry more than it conveys, and the gate cannot pass more
        than its orifice lets through at the head it works under. Whichever
        is smaller is the one that binds; on the upper reaches of this
        canal that is the gate.

        The lower bound is the same in both readings - a gate cannot pass
        less than nothing, so the deviation cannot go below minus the
        nominal discharge.
        """
        gates = self.gate_capacity_m3_s
        if gates is None:
            gates = self.capacity_m3_s
        return tuple(
            (-nominal, min(capacity, gate) - nominal)
            for capacity, gate, nominal in zip(
                self.capacity_m3_s, gates, self.nominal_m3_s
            )
        )

    @property
    def command_bounds(self) -> tuple[tuple[float, float], ...]:
        """C5': the region in which the linear model is the model.

        Written on what the controller *asks* for, not on what the filter
        lets through, and it is a different constraint from C5 rather than
        a restatement of it. The command swings several times harder than
        the applied flow - on this canal between three and eight times per
        unit of order - so a schedule that keeps the applied flow inside
        the conveyance can still demand a command far outside the range the
        pool models were identified over, and outside it the answer is not
        a statement about this canal at all.
        """
        return tuple(
            (-nominal, capacity - nominal)
            for capacity, nominal in zip(self.capacity_m3_s, self.nominal_m3_s)
        )


@dataclass(frozen=True)
class Scenario:
    """A whole experiment: the canal, the users, the limits and the water.

    ``source_discharge_m3_s`` is absolute - what the headworks can release -
    and the scan the experiments run is over this number as a fraction of
    the aggregate demand.
    """

    name: str
    network: Network
    users: tuple[User, ...]
    blocks: int
    limits: Limits
    source_discharge_m3_s: float
    source_nominal_until: int = 0
    source_ramp_steps: int = STEPS_PER_BLOCK
    loss_factor: tuple[float, ...] | None = None
    demand_floor_m3: float = D_MIN_M3
    ratio_cap: float = R_CAP
    steps_per_block: int = STEPS_PER_BLOCK
    dt_s: float = DT_PLANT_S
    #: Settling steps after the last order block. The floor under it is the
    #: filter's memory, so it belongs to the scenario and not to the plant:
    #: a response map and a scenario that disagree about the horizon cannot
    #: be assembled into one programme, and lowering the filter's cut-off
    #: lengthens the memory past the frozen value.
    settle_margin: int = SETTLE_MARGIN_STEPS
    excluded: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.users:
            raise ScenarioError(f"{self.name}: a scenario needs at least one user")
        if self.blocks < 1:
            raise ScenarioError(f"{self.name}: a scenario needs at least one block")
        if self.limits.size != self.network.size:
            raise ScenarioError(
                f"{self.name}: limits for {self.limits.size} reaches but the network "
                f"has {self.network.size}"
            )
        if self.source_discharge_m3_s <= 0.0:
            raise ScenarioError(f"{self.name}: the source must release something")
        if self.source_ramp_steps < 0:
            raise ScenarioError(f"{self.name}: a ramp cannot be negative")
        if not 0 <= self.source_nominal_until <= self.horizon:
            raise ScenarioError(
                f"{self.name}: the restriction starts at step "
                f"{self.source_nominal_until}, outside the run"
            )
        names = [user.name for user in self.users]
        if len(set(names)) != len(names):
            raise ScenarioError(f"{self.name}: user names must be unique")

        steps = self.blocks * self.steps_per_block
        for user in self.users:
            if user.node > self.network.size:
                raise ScenarioError(
                    f"{user.name}: node {user.node} is outside the network"
                )
            if user.announced_block >= self.blocks:
                raise ScenarioError(
                    f"{user.name}: announced at block {user.announced_block} of "
                    f"{self.blocks}, so it never gets to order"
                )
            if user.demand_m3 < self.demand_floor_m3:
                raise ScenarioError(
                    f"{user.name}: a demand of {user.demand_m3:.3f} m^3 is below the "
                    f"declared floor of {self.demand_floor_m3:.3f}; exclude it "
                    f"explicitly and record it, rather than letting a small "
                    f"denominator decide the lexicographic order"
                )
            if user.window[1] >= self.horizon:
                raise ScenarioError(
                    f"{user.name}: its window ends at step {user.window[1]}, past the "
                    f"run's last step {self.horizon - 1}"
                )
            if user.window[0] >= steps:
                raise ScenarioError(
                    f"{user.name}: its window starts at step {user.window[0]}, after "
                    f"the last block ends at {steps - 1}, so nothing it orders can "
                    f"reach it in time"
                )
        if self.loss_factor is not None and len(self.loss_factor) != self.network.size:
            raise ScenarioError(
                f"{self.name}: a loss factor per reach, or none at all"
            )

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def horizon(self) -> int:
        from faircanal.plant import horizon_for

        return horizon_for(self.blocks, self.steps_per_block, self.settle_margin)

    @property
    def aggregate_demand_m3(self) -> float:
        return sum(user.demand_m3 for user in self.users)

    @property
    def nominal_draw_m3_s(self) -> float:
        """Total flow the users draw when nobody orders anything."""
        return sum(user.nominal_m3_s for user in self.users)

    @property
    def scarcity(self) -> float:
        """Source discharge as a fraction of what the users draw nominally.

        One at the published operating point, below one when the canal is
        short. This is the scan variable of the experiments, and it is
        defined here so that no run can report a scarcity level that was
        computed some other way.
        """
        return self.source_discharge_m3_s / self.nominal_draw_m3_s

    def source_profile(self) -> "list[float]":
        """What the source may release at each step of the run.

        The nominal draw until the restriction takes effect, the declared
        discharge until the last block ends, and the nominal draw again
        while the last order is still arriving - see the module docstring
        for why the last part is not optional.
        """
        nominal = self.nominal_draw_m3_s
        restricted = self.source_discharge_m3_s
        start = self.source_nominal_until
        last = self.blocks * self.steps_per_block
        ramp = max(1, self.source_ramp_steps)
        profile = []
        for step in range(self.horizon):
            if step < start or step >= last:
                profile.append(nominal)
            elif step < start + ramp:
                share = (step - start + 1) / ramp
                profile.append(nominal + share * (restricted - nominal))
            elif step >= last - ramp:
                share = (last - step) / ramp
                profile.append(nominal + share * (restricted - nominal))
            else:
                profile.append(restricted)
        return profile

    def users_at(self, node: int) -> tuple[int, ...]:
        """Indices of the users drawing at one gate, in order."""
        return tuple(
            index for index, user in enumerate(self.users) if user.node == node
        )


def one_user_per_gate(
    network: Network,
    blocks: int,
    limits: Limits,
    source_discharge_m3_s: float,
    window: "tuple[int, int] | None" = None,
    demand_scale: float = 1.0,
    lead_blocks: int = 0,
    announced_block: int = ANNOUNCE_BLOCKS,
    outlet_headroom: float = OUTLET_HEADROOM,
    overshoot: float = 0.0,
    steps_per_block: int = STEPS_PER_BLOCK,
    dt_s: float = DT_PLANT_S,
    settle_margin: int = SETTLE_MARGIN_STEPS,
    name: str = "one-per-gate",
) -> Scenario:
    """The simplest scenario the published data supports: one user per gate.

    Each gate's published offtake becomes one user, whose nominal draw is
    that offtake and whose demand is ``demand_scale`` times what it would
    receive across the window by drawing nominally. So ``demand_scale = 1``
    is "everyone asks for their usual amount" - the case in which ordering
    nothing is already a perfect score, and every departure from one is the
    filter and the timing rather than the water.

    ``lead_blocks`` is how long the canal runs undisturbed before the
    restriction takes effect and the delivery window opens. It exists
    because a filtered order cannot change the flow at the first sample,
    so a run whose window and restriction both start at step zero is
    infeasible for every schedule - see the module docstring. Zero is
    allowed and is what the unrestricted case uses.

    The window otherwise runs to the end, which is the least restrictive
    choice and therefore the one that hides the least.

    ``overshoot`` is the delivery allowance of the volume budget C3: how
    much more than the demand a user may be released, as a fraction. Zero
    is the study's own setting and the one every headline number uses. It
    is a parameter here because one of the pre-registered questions is
    whether the method's advantage comes from the criterion or simply from
    letting more water out, and the only way to answer that is to run the
    same scan with the budget loosened and see whether anything moves.
    """
    from faircanal.plant import horizon_for

    horizon = horizon_for(blocks, steps_per_block, settle_margin)
    lead = lead_blocks * steps_per_block
    if lead_blocks and announced_block >= lead_blocks:
        raise ScenarioError(
            f"orders may not be re-shaped before block {announced_block}, but the "
            f"restriction takes effect at block {lead_blocks}: the users would be "
            f"told about the shortage at the moment it arrived. With a filter "
            f"between an order and the water, no schedule exists, and the "
            f"infeasibility would read as a finding about the canal rather than "
            f"about the notice"
        )
    if window is None:
        window = (lead, horizon - 1)
    users = []
    for reach in network.reaches:
        if reach.offtake_m3_s <= 0.0:
            continue
        steps = window[1] - window[0] + 1
        users.append(
            User(
                name=f"{reach.label}-user",
                node=reach.node,
                nominal_m3_s=reach.offtake_m3_s,
                demand_m3=demand_scale * reach.offtake_m3_s * steps * dt_s,
                window=window,
                announced_block=announced_block,
                max_order_m3_s=outlet_headroom * reach.offtake_m3_s,
                overshoot=overshoot,
            )
        )
    return Scenario(
        name=name,
        network=network,
        users=tuple(users),
        blocks=blocks,
        limits=limits,
        source_discharge_m3_s=source_discharge_m3_s,
        source_nominal_until=lead,
        steps_per_block=steps_per_block,
        dt_s=dt_s,
        settle_margin=settle_margin,
    )
