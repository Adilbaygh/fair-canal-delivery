"""Canal geometry and the low-order models identified from it.

The experiments of this study run on the ASCE test canals rather than on
the reach models of the source control study, and the two parameter sets
are never mixed. This module holds the geometry, the textbook hydraulics
that turn it into flows and depths, and the integrator-delay-zero
quantities the pool models are built from.

Why the zero matters here
-------------------------
The whole study is about a low-pass filter, and a filter exists to
suppress something. Clemmens et al. say why in one sentence: the filter is
used "because the wave created by gate movements can cause the controller
to be unstable if not filtered". A canal model with no wave in it would
make the filter unmotivated and the study with it, so the pool models
carry the first resonance and the geometry says where it sits.

Which canal is the primary one, and why
---------------------------------------
Corning, the larger and flatter of the two. The same paper reports of the
smaller canal that "pools in this canal do not experience oscillations,
except in the first pool" - which makes it the wrong place to study a
wave-suppressing filter. It is kept as a second case, and its lack of
oscillation is reported rather than hidden: it shows where the method is
needed and where it is not.

Sources
-------
Corning geometry and flows: Bonet E, Yubero MT, Bascompta M, Alfonso P
(2025). A Linear Model for Irrigation Canals Operating in Real Time
Applied in ASCE Test Cases. Water 17(9):1368, doi:10.3390/w17091368,
CC BY. Tables 5 and 6.

WM lateral geometry, flows and the integrator-delay-zero relations:
Clemmens AJ, Tian X, van Overloop P-J, Litrico X (2015). Integrator Delay
Zero Model for Design of Upstream Water-Level Controllers. J. Irrig.
Drain. Eng., paper B4015001,
doi:10.1061/(ASCE)IR.1943-4774.0000997. Table 1 and Eqs (2), (3), (15),
and Eqs (24)-(31) for the reflection waves.

Target depths of both canals, and an independent statement of both
geometries: Litrico X, Fromion V (2004). Simplified Modeling of Irrigation
Canals for Controller Design. J. Irrig. Drain. Eng. 130(5):373-383,
doi:10.1061/(ASCE)0733-9437(2004)130:5(373). Tables 1 and 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

__all__ = [
    "GRAVITY",
    "TrapezoidalPool",
    "Gate",
    "CORNING_POOLS",
    "CORNING_GATES",
    "CORNING_OFFTAKES",
    "CORNING_CHECK_FLOWS",
    "CORNING_HEADING_FLOW",
    "WM_POOLS",
    "WM_GATES",
    "WM_CHECK_FLOWS",
    "WM_OFFTAKES",
    "WM_HEADING_FLOW",
    "top_width",
    "flow_area",
    "wetted_perimeter",
    "hydraulic_radius",
    "hydraulic_depth",
    "wave_celerity",
    "uniform_discharge",
    "normal_depth",
    "backwater_area",
    "resonance_frequency",
    "resonance_peak_height",
    "wave_decay_rate",
    "reflection_factor",
    "resonance_peak_with_reflections",
    "gate_capacity",
    "LITRICO_GEOMETRY_SOURCE",
]

#: Standard gravity [m/s^2].
GRAVITY = 9.81


@dataclass(frozen=True)
class TrapezoidalPool:
    """One reach of a trapezoidal canal, as its designers specified it.

    ``side_slope`` is horizontal over vertical, so a value of 1.5 means the
    bank rises one metre for every one and a half metres out.
    """

    name: str
    length_m: float
    bed_width_m: float
    side_slope: float
    canal_depth_m: float
    manning_n: float
    bed_slope: float
    target_level_m: float | None = None
    provenance: str = "observed"
    source: str = ""

    def __post_init__(self) -> None:
        if self.length_m <= 0.0 or self.bed_width_m <= 0.0:
            raise ValueError(f"{self.name}: length and bed width must be positive")
        if self.side_slope < 0.0:
            raise ValueError(f"{self.name}: side slope cannot be negative")
        if self.manning_n <= 0.0 or self.bed_slope <= 0.0:
            raise ValueError(f"{self.name}: roughness and bed slope must be positive")


@dataclass(frozen=True)
class Gate:
    """A check gate, described by what limits the flow through it."""

    name: str
    width_m: float
    height_m: float
    discharge_coefficient: float = 0.61
    provenance: str = "observed"
    source: str = ""

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError(f"{self.name}: gate width and height must be positive")
        if not 0.0 < self.discharge_coefficient <= 1.0:
            raise ValueError(f"{self.name}: implausible discharge coefficient")


# ---------------------------------------------------------------------------
# Textbook hydraulics
# ---------------------------------------------------------------------------


def top_width(pool: TrapezoidalPool, depth: float) -> float:
    """Water-surface width at *depth* [m]."""
    return pool.bed_width_m + 2.0 * pool.side_slope * depth


def flow_area(pool: TrapezoidalPool, depth: float) -> float:
    """Cross-sectional area of flow at *depth* [m^2]."""
    return (pool.bed_width_m + pool.side_slope * depth) * depth


def wetted_perimeter(pool: TrapezoidalPool, depth: float) -> float:
    """Length of the wetted boundary at *depth* [m]."""
    return pool.bed_width_m + 2.0 * depth * math.hypot(1.0, pool.side_slope)


def hydraulic_radius(pool: TrapezoidalPool, depth: float) -> float:
    """Area divided by wetted perimeter [m]."""
    return flow_area(pool, depth) / wetted_perimeter(pool, depth)


def hydraulic_depth(pool: TrapezoidalPool, depth: float) -> float:
    """Area divided by top width [m].

    This, and not the depth itself, is what sets the wave speed. In a wide
    shallow channel the two nearly coincide; in a narrow trapezoidal one
    they do not, and using the wrong one moves the resonance.
    """
    return flow_area(pool, depth) / top_width(pool, depth)


def wave_celerity(pool: TrapezoidalPool, depth: float) -> float:
    """Speed of a gravity wave on the surface [m/s], ``sqrt(g D)``."""
    return math.sqrt(GRAVITY * hydraulic_depth(pool, depth))


def uniform_discharge(pool: TrapezoidalPool, depth: float) -> float:
    """Manning's discharge for uniform flow at *depth* [m^3/s]."""
    if depth <= 0.0:
        return 0.0
    return (
        flow_area(pool, depth)
        * hydraulic_radius(pool, depth) ** (2.0 / 3.0)
        * math.sqrt(pool.bed_slope)
        / pool.manning_n
    )


def normal_depth(pool: TrapezoidalPool, discharge: float) -> float:
    """Depth at which uniform flow carries *discharge* [m].

    Inverts :func:`uniform_discharge` numerically. Discharge rises
    monotonically with depth in a trapezoidal channel, so the root is
    unique and bracketing is safe.
    """
    if discharge < 0.0:
        raise ValueError(f"{pool.name}: discharge cannot be negative")
    if discharge == 0.0:
        return 0.0

    upper = max(pool.canal_depth_m, 1.0)
    while uniform_discharge(pool, upper) < discharge:
        upper *= 2.0
        if upper > 1.0e4:
            raise ValueError(
                f"{pool.name}: no normal depth below 10 km for {discharge} m^3/s"
            )
    return float(brentq(lambda y: uniform_discharge(pool, y) - discharge, 1e-9, upper))


# ---------------------------------------------------------------------------
# The integrator-delay-zero quantities
# ---------------------------------------------------------------------------


def backwater_area(pool: TrapezoidalPool, depth: float) -> float:
    """Storage area of a reach held under backwater [m^2].

    Equation (2) of the source: the top width times the pool length. The
    paper is explicit that this is an approximation and that a reach not
    under backwater over its whole length has a much smaller area, so a
    value from it is labelled ``derived`` and compared against the tuned
    values the same paper publishes.
    """
    return top_width(pool, depth) * pool.length_m


def resonance_frequency(
    pool: TrapezoidalPool, depth: float, velocity: float
) -> float:
    """Frequency of the first resonance [rad/s].

    Equation (15): the wave runs the length of the pool and back, downstream
    at ``c + v`` and upstream at ``c - v``, and the round trip sets the
    period.
    """
    celerity = wave_celerity(pool, depth)
    if velocity >= celerity:
        raise ValueError(
            f"{pool.name}: flow at {velocity} m/s is supercritical against a "
            f"celerity of {celerity:.3f} m/s; the wave cannot travel upstream "
            f"and this model does not apply"
        )
    cycle = pool.length_m * (1.0 / (celerity + velocity) + 1.0 / (celerity - velocity))
    return 2.0 * math.pi / cycle


def resonance_peak_height(
    pool: TrapezoidalPool, depth: float, velocity: float
) -> float:
    """Height of the resonance peak [s/m^2].

    Equation (3): ``1 / (B (c - v))``. The paper notes this is a
    conservative estimate when the downstream depth is at or near normal
    depth, and that a reach well under backwater can peak higher.
    """
    celerity = wave_celerity(pool, depth)
    if velocity >= celerity:
        raise ValueError(
            f"{pool.name}: flow at {velocity} m/s is supercritical against a "
            f"celerity of {celerity:.3f} m/s"
        )
    return 1.0 / (top_width(pool, depth) * (celerity - velocity))


def shape_exponent(pool: TrapezoidalPool, depth: float) -> float:
    """The friction exponent ``kappa`` of the linearised flow equations.

    Equation (29) of Clemmens et al.: ``7/3 - (4 A)/(3 B P) dP/dy``. The
    same quantity appears as ``kappa_0`` in Eq. (34) of Litrico and
    Fromion, written with ``T_0`` in place of ``B``; the two definitions
    are the same expression, which is one of several places where the two
    papers can be checked against each other.

    For a trapezoidal channel ``dP/dy = 2 sqrt(1 + m^2)`` with ``m`` the
    side slope, which is a constant.
    """
    area = flow_area(pool, depth)
    width = top_width(pool, depth)
    perimeter = wetted_perimeter(pool, depth)
    dp_dy = 2.0 * math.hypot(1.0, pool.side_slope)
    return 7.0 / 3.0 - 4.0 * area / (3.0 * width * perimeter) * dp_dy


def wave_decay_rate(pool: TrapezoidalPool, depth: float, velocity: float) -> float:
    """How fast a gravity wave loses height as it travels, per metre.

    Equations (25)-(29) of Clemmens et al.: ``r = r1 + r2``, the sum of the
    rates for the downstream and the upstream leg, so that a wave that runs
    the pool and comes back is multiplied by ``exp(-r L)``.

    That single number decides whether a pool oscillates. The paper reports
    of the steep test canal that its pools "do not experience oscillations,
    except in the first pool", and this function reproduces that: the first
    pool keeps about eleven per cent of a wave over a round trip and the
    rest keep almost none - see
    ``test_only_the_first_pool_of_the_steep_canal_rings``.

    This is written for uniform flow, which is the condition the source's
    appendix states for it.
    """
    celerity = wave_celerity(pool, depth)
    if velocity <= 0.0:
        raise ValueError(f"{pool.name}: the decay rate is undefined at zero flow")
    if velocity >= celerity:
        raise ValueError(
            f"{pool.name}: flow at {velocity} m/s is not subcritical against a "
            f"celerity of {celerity:.3f} m/s"
        )
    down = celerity + velocity  # alpha in the source
    up = celerity - velocity  # beta in the source
    gamma = GRAVITY * (1.0 + shape_exponent(pool, depth)) * pool.bed_slope
    delta = 2.0 * GRAVITY * pool.bed_slope / velocity
    r1 = (down * delta - gamma) / (down * (down + up))
    r2 = (up * delta + gamma) / (up * (down + up))
    return r1 + r2


def reflection_factor(pool: TrapezoidalPool, depth: float, velocity: float) -> float:
    """How much the reflected waves raise the resonance peak, as a ratio.

    Dividing Eq. (31) by Eq. (3) leaves

        (1 + (c+v)/(c-v) exp(-r L)) / (1 - exp(-r L)),

    a dimensionless number that is one when nothing comes back and grows
    without bound as the damping vanishes. Writing it as a ratio rather
    than as a peak height is what lets the same amplification be applied to
    a peak whose base is taken from a different source, which is what
    :mod:`faircanal.identify` does.

    Reading the source's Eq. (31): the printed numerator and denominator
    are both perfect squares - ``1 + q^2 e^{-2rL} + 2 q e^{-rL}`` and
    ``1 + e^{-2rL} - 2 e^{-rL}`` - so the square root of their ratio is
    exactly the expression above. That is also what makes Eq. (30) a
    maximum at ``cos(w t) = 1``, which the source states.
    """
    decay = math.exp(-wave_decay_rate(pool, depth, velocity) * pool.length_m)
    celerity = wave_celerity(pool, depth)
    ratio = (celerity + velocity) / (celerity - velocity)
    return (1.0 + ratio * decay) / (1.0 - decay)


def resonance_peak_with_reflections(
    pool: TrapezoidalPool, depth: float, velocity: float
) -> float:
    """Height of the resonance peak with the reflected waves, [s/m^2].

    Equation (31), assembled as Eq. (3) times :func:`reflection_factor`.
    """
    return resonance_peak_height(pool, depth, velocity) * reflection_factor(
        pool, depth, velocity
    )


def gate_capacity(gate: Gate, head_difference_m: float) -> float:
    """Largest discharge the gate can pass at a given head drop [m^3/s].

    The orifice relation of the model document: ``C_d W H sqrt(2 g dH)``.
    The gate equation itself stays outside the optimisation; only this
    bound enters it, which is what keeps the programme linear.
    """
    if head_difference_m < 0.0:
        raise ValueError(f"{gate.name}: head difference cannot be negative")
    return (
        gate.discharge_coefficient
        * gate.width_m
        * gate.height_m
        * math.sqrt(2.0 * GRAVITY * head_difference_m)
    )


# ---------------------------------------------------------------------------
# ASCE Test Canal 2 - the Corning canal, California
# ---------------------------------------------------------------------------

#: Litrico and Fromion publish both ASCE canals independently of the two
#: sources used above, and every length, bed width, side slope, roughness
#: and bed slope agrees with them exactly - which is why the target depths
#: below can be taken from that paper without mixing parameter sets. It is
#: also the only published statement of Corning's target depths found for
#: this study; ``test_the_two_published_geometries_agree`` keeps the
#: agreement from being assumed rather than checked.
LITRICO_GEOMETRY_SOURCE = (
    "Litrico X, Fromion V (2004). Simplified Modeling of Irrigation Canals "
    "for Controller Design. J. Irrig. Drain. Eng. 130(5):373-383, "
    "doi:10.1061/(ASCE)0733-9437(2004)130:5(373). Table 1 for ASCE test "
    "canal 1 and Table 2 for ASCE test canal 2, columns X, B and Y_X."
)

_CORNING_SOURCE = (
    "Bonet E, Yubero MT, Bascompta M, Alfonso P (2025). A Linear Model for "
    "Irrigation Canals Operating in Real Time Applied in ASCE Test Cases. "
    "Water 17(9):1368, doi:10.3390/w17091368, CC BY. Tables 5 and 6. "
    "Target depths from " + LITRICO_GEOMETRY_SOURCE
)

#: Eight reaches, most upstream first, as Table 5 lists them.
CORNING_POOLS = tuple(
    TrapezoidalPool(
        name=f"corning-{index}",
        length_m=length * 1000.0,
        bed_width_m=bed_width,
        side_slope=1.5,
        canal_depth_m=canal_depth,
        manning_n=0.02,
        bed_slope=1.0e-4,
        target_level_m=target,
        source=_CORNING_SOURCE,
    )
    for index, (length, bed_width, canal_depth, target) in enumerate(
        (
            (7.0, 7.0, 2.5, 2.1),
            (3.0, 7.0, 2.5, 2.1),
            (3.0, 7.0, 2.5, 2.1),
            (4.0, 6.0, 2.3, 1.9),
            (4.0, 6.0, 2.3, 1.9),
            (3.0, 5.0, 2.3, 1.7),
            (2.0, 5.0, 1.9, 1.7),
            (2.0, 5.0, 1.9, 1.7),
        ),
        start=1,
    )
)

#: Check gates. The eighth reach ends the canal and has none.
CORNING_GATES = tuple(
    Gate(name=f"corning-gate-{index}", width_m=width, height_m=height, source=_CORNING_SOURCE)
    for index, (width, height) in enumerate(
        ((7.0, 2.3), (7.0, 2.3), (7.0, 2.3), (6.0, 2.1), (6.0, 2.1), (5.0, 1.8), (5.0, 1.8)),
        start=1,
    )
)

#: Initial offtake discharges, Table 6 [m^3/s].
CORNING_OFFTAKES = (1.7, 1.8, 2.7, 0.3, 0.2, 0.8, 1.2, 2.3)

#: Initial discharge at each check, Table 6 [m^3/s].
CORNING_CHECK_FLOWS = (12.0, 10.2, 7.5, 7.2, 7.0, 6.2, 5.0, 2.7)

#: Discharge entering the canal at its head, Table 6 [m^3/s].
CORNING_HEADING_FLOW = 13.7


# ---------------------------------------------------------------------------
# ASCE Test Canal 1 - the WM lateral, Maricopa-Stanfield, Arizona
# ---------------------------------------------------------------------------

_WM_SOURCE = (
    "Clemmens AJ, Tian X, van Overloop P-J, Litrico X (2015). Integrator Delay "
    "Zero Model for Design of Upstream Water-Level Controllers. J. Irrig. "
    "Drain. Eng., paper B4015001, doi:10.1061/(ASCE)IR.1943-4774.0000997, "
    "Table 1. Bed widths are given here and not in the other published "
    "description of this canal."
)

WM_POOLS = tuple(
    TrapezoidalPool(
        name=f"wm-{index}",
        length_m=length,
        bed_width_m=bed_width,
        side_slope=1.5,
        canal_depth_m=canal_depth,
        manning_n=0.014,
        bed_slope=0.002,
        target_level_m=target,
        source=_WM_SOURCE,
    )
    for index, (length, bed_width, canal_depth, target) in enumerate(
        (
            (100.0, 1.0, 1.1, 0.9),
            (1200.0, 1.0, 1.1, 0.9),
            (400.0, 1.0, 1.0, 0.8),
            (800.0, 0.8, 1.1, 0.9),
            (2000.0, 0.8, 1.1, 0.9),
            (1700.0, 0.8, 1.0, 0.8),
            (1600.0, 0.6, 1.0, 0.8),
            (1700.0, 0.6, 1.0, 0.8),
        ),
        start=1,
    )
)

WM_GATES = tuple(
    Gate(name=f"wm-gate-{index}", width_m=width, height_m=height, source=_WM_SOURCE)
    for index, (width, height) in enumerate(
        ((1.5, 1.0), (1.5, 1.0), (1.5, 0.9), (1.2, 1.0), (1.2, 1.0), (1.2, 0.9), (1.0, 0.9)),
        start=1,
    )
)

#: Test 1-1 initial discharge at each check [m^3/s], Table 1's last column.
#:
#: These are the flows past each check gate, not the offtakes. They fall by
#: exactly one tenth of a cubic metre per second per reach, which is the
#: offtake at each one - stated independently in the other published
#: description of this canal as "the offtakes at the end of each pool are
#: set to 0.1 m^3/s throughout the test".
WM_CHECK_FLOWS = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)

#: Offtake at the end of each reach for Test 1-1 [m^3/s], derived from the
#: check flows above and confirmed by the scenario description.
WM_OFFTAKES = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

#: Discharge at the headgate for Test 1-1 [m^3/s].
WM_HEADING_FLOW = 0.8
