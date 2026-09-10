"""Integrator-delay-zero parameters, computed from geometry and checked
against the published tables.

Why this module has to exist
----------------------------
The experiments of this study run the canal at flows that nobody has
published a pool model for: the whole point is to scan the head discharge
from full supply down to well below aggregate demand and watch what the
allocation does. A table of coefficients cannot follow the canal down that
scan. So the pool models are computed from the geometry, and the published
tables become the thing that says the computation is right.

The two routes
--------------
Both routes are implemented and both are kept:

* :func:`idz_from_geometry` follows Litrico and Fromion's Appendices I-IV
  from the trapezoidal section, the bed slope, the roughness, the
  discharge and the target depth;
* :data:`LITRICO_TABLES` holds what the same paper printed for the two ASCE
  test canals at two flows each - thirty-two pools in all.

``test_the_geometry_route_reproduces_the_published_tables`` runs one
against the other, and the agreement it finds is what the manuscript
reports; nothing here is asserted without it.

What the model is
-----------------
Equations (3) and (4) of the source, for the downstream end of a pool:

    A_d dh/dt = q(0, t - tau_d) - q(X, t)
    y(X, t)   = h(t) + p21 q(0, t - tau_d) - p22 q(X, t)

An integrator whose gain is set by a backwater area, a transport delay on
the inflow, and a direct feedthrough at each end. The oscillating modes are
deliberately absent - "these modes are usually filtered, in order not to
destabilize the controller", which is this study's low-pass filter stated
by a different author for a different reason. :mod:`faircanal.identify`
puts the first of those modes back, because a study about a filter needs
the thing the filter removes.

One reading had to be settled
------------------------------
Equation (41) prints the water surface of the backed-up part as a straight
line of slope ``S_b``, the bed slope, while the sentence under it says
"the water level is a straight line of slope ``S_X``" and Eq. (42) is only
consistent with ``S_X``. Both readings are implemented; ``S_X`` reproduces
the published tables to a twentieth of a per cent on the flat canal and
``S_b`` misses them by up to fifty-four per cent, so the question is
settled by measurement rather than by argument. See
``test_the_bed_slope_reading_of_the_backwater_profile_is_refuted``.

Source
------
Litrico X, Fromion V (2004). Simplified Modeling of Irrigation Canals for
Controller Design. J. Irrig. Drain. Eng. 130(5):373-383,
doi:10.1061/(ASCE)0733-9437(2004)130:5(373). Model: Eqs (3), (4), (11),
(12), (43)-(46). Construction: Eqs (32)-(42) and (47)-(66). Published
parameters: Tables 1-8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from faircanal.geometry import (
    GRAVITY,
    TrapezoidalPool,
    flow_area,
    hydraulic_radius,
    normal_depth,
    shape_exponent,
    top_width,
    wave_celerity,
)

__all__ = [
    "LITRICO_SOURCE",
    "IdzParams",
    "LitricoRow",
    "LITRICO_TABLES",
    "LITRICO_MAX_RELATIVE_ERRORS",
    "PROFILE_SLOPE_READINGS",
    "friction_slope",
    "froude_number",
    "depth_gradient",
    "BackwaterSplit",
    "backwater_split",
    "idz_from_geometry",
    "round_trip_decay",
    "reflection_amplification",
    "decay_implied_by_published_range",
]

LITRICO_SOURCE = (
    "Litrico X, Fromion V (2004). Simplified Modeling of Irrigation Canals "
    "for Controller Design. J. Irrig. Drain. Eng. 130(5):373-383, "
    "doi:10.1061/(ASCE)0733-9437(2004)130:5(373)."
)

#: The two readings of Eq. (41). See the module docstring.
PROFILE_SLOPE_READINGS = ("surface", "bed")


@dataclass(frozen=True)
class IdzParams:
    """The four numbers that describe one pool at one operating point.

    ``delay_s`` and ``area_m2`` are ``tau_d`` and ``A_d`` of Eq. (4);
    ``p21`` and ``p22`` are the high-frequency gains of Eqs (45) and (46),
    in seconds per square metre. ``p21`` multiplies the inflow, which
    arrives delayed, and ``p22`` the discharge leaving at the downstream
    end, which is felt at once.
    """

    name: str
    discharge_m3_s: float
    depth_m: float
    delay_s: float
    area_m2: float
    p21: float
    p22: float
    units: str = "s, m^2, s/m^2"
    provenance: str = "derived"
    source: str = LITRICO_SOURCE

    def __post_init__(self) -> None:
        if self.delay_s < 0.0:
            raise ValueError(f"{self.name}: a transport delay cannot be negative")
        if self.area_m2 <= 0.0:
            raise ValueError(f"{self.name}: the backwater area must be positive")
        if self.p21 < 0.0 or self.p22 <= 0.0:
            raise ValueError(f"{self.name}: the high-frequency gains must be positive")


# ---------------------------------------------------------------------------
# What the source printed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LitricoRow:
    """One row of Tables 4 to 7, transcribed as printed.

    Both the theoretical value and the model's approximation of it are kept
    for the delay and the area, because the gap between them is the source's
    own statement of how good the model is - Table 8 - and it is the yardstick
    the geometry route has to be judged against.
    """

    pool: int
    discharge_m3_s: float
    delay_theory_s: float
    delay_model_s: float
    area_theory_m2: float
    area_model_m2: float
    p21_range: tuple[float, float]
    p21: float
    p22_range: tuple[float, float]
    p22: float


def _rows(raw) -> tuple[LitricoRow, ...]:
    return tuple(
        LitricoRow(
            pool=index,
            discharge_m3_s=entry[0],
            delay_theory_s=entry[1],
            delay_model_s=entry[2],
            area_theory_m2=entry[3],
            area_model_m2=entry[4],
            p21_range=(entry[5], entry[6]),
            p21=entry[7],
            p22_range=(entry[8], entry[9]),
            p22=entry[10],
        )
        for index, entry in enumerate(raw, start=1)
    )


#: Tables 4-7, keyed by ``(canal, flow)``. Canal 1 is the steep WM lateral
#: and canal 2 is the flat Corning canal, in the source's own numbering.
LITRICO_TABLES = {
    ("wm", "low"): _rows(
        (
            (0.8, 36.9, 36.9, 341.6, 339.2, 0.1300, 1.4069, 0.1913, 0.0295, 1.2644, 0.1391),
            (0.7, 426.4, 427.6, 821.0, 824.9, 0.0054, 0.0087, 0.0070, 0.0880, 0.2039, 0.2368),
            (0.6, 151.9, 152.9, 668.7, 671.2, 0.1092, 0.1737, 0.1440, 0.1014, 0.2253, 0.2719),
            (0.5, 305.9, 307.6, 788.5, 790.8, 0.0219, 0.0343, 0.0274, 0.0993, 0.2214, 0.2561),
            (0.4, 792.9, 794.9, 817.2, 817.8, 4.2e-5, 6.2e-5, 5.3e-5, 0.0948, 0.1967, 0.2565),
            (0.3, 723.0, 724.8, 679.5, 678.3, 6.1e-5, 9.9e-5, 7.6e-5, 0.1014, 0.2631, 0.2951),
            (0.2, 732.1, 734.2, 635.5, 632.7, 2.5e-5, 4.0e-5, 3.3e-5, 0.1033, 0.2534, 0.3289),
            (0.1, 912.3, 913.4, 673.9, 667.1, 2.5e-7, 4.4e-7, 3.5e-7, 0.0862, 0.2691, 0.3394),
        )
    ),
    ("wm", "high"): _rows(
        (
            (2.0, 29.5, 29.6, 328.2, 331.2, 0.1598, 0.6165, 0.2454, 0.0880, 0.5698, 0.1974),
            (1.8, 337.6, 337.4, 574.5, 575.9, 0.0168, 0.0251, 0.0208, 0.1311, 0.2373, 0.2854),
            (1.6, 117.2, 117.0, 438.3, 436.3, 0.1188, 0.1760, 0.1477, 0.1572, 0.2793, 0.3451),
            (1.4, 238.4, 238.2, 572.6, 575.8, 0.0387, 0.0602, 0.0490, 0.1281, 0.2522, 0.2916),
            (1.2, 610.8, 610.8, 616.4, 620.8, 0.0011, 0.0016, 0.0013, 0.1229, 0.2479, 0.2791),
            (1.0, 541.3, 541.3, 494.3, 496.2, 0.0018, 0.0028, 0.0022, 0.1502, 0.2879, 0.3260),
            (0.8, 531.0, 531.0, 465.3, 467.6, 0.0015, 0.0024, 0.0020, 0.1478, 0.2868, 0.3492),
            (0.6, 603.0, 603.5, 516.0, 518.8, 5.6e-4, 7.7e-4, 6.9e-4, 0.1452, 0.2501, 0.3337),
        )
    ),
    ("corning", "low"): _rows(
        (
            (2.7, 1822.6, 1821.4, 82799, 83746, 0.0164, 0.0230, 0.0246, 0.0142, 0.0273, 0.0225),
            (2.5, 754.6, 754.4, 38199, 38247, 0.0198, 0.0785, 0.0260, 0.0054, 0.0757, 0.0204),
            (2.2, 757.7, 757.5, 38278, 38315, 0.0199, 0.0884, 0.0263, 0.0048, 0.0852, 0.0203),
            (2.0, 1073.4, 1073.0, 43607, 43767, 0.0230, 0.0518, 0.0302, 0.0108, 0.0521, 0.0254),
            (1.7, 1080.0, 1079.5, 43822, 43941, 0.0235, 0.0613, 0.0310, 0.0093, 0.0596, 0.0253),
            (1.5, 848.8, 848.7, 28481, 28560, 0.0285, 0.0725, 0.0370, 0.0119, 0.0722, 0.0308),
            (1.2, 564.1, 564.0, 19476, 19490, 0.0291, 0.1457, 0.0387, 0.0063, 0.1408, 0.0298),
            (1.0, 566.9, 566.8, 19514, 19523, 0.0293, 0.1751, 0.0392, 0.0053, 0.1686, 0.0297),
        )
    ),
    ("corning", "high"): _rows(
        (
            (11.0, 1577.9, 1582.8, 59703, 61245, 0.0052, 0.0053, 0.0061, 0.0216, 0.0221, 0.0240),
            (10.0, 681.5, 682.0, 33868, 34113, 0.0158, 0.0215, 0.0185, 0.0164, 0.0285, 0.0216),
            (9.0, 690.4, 690.9, 34630, 34891, 0.0163, 0.0235, 0.0195, 0.0154, 0.0296, 0.0214),
            (8.0, 952.4, 953.7, 35441, 35885, 0.0127, 0.0139, 0.0154, 0.0241, 0.0282, 0.0270),
            (7.0, 970.3, 971.9, 37029, 37598, 0.0139, 0.0156, 0.0177, 0.0231, 0.0284, 0.0268),
            (6.0, 753.7, 754.3, 23458, 23634, 0.0176, 0.0198, 0.0200, 0.0291, 0.0360, 0.0328),
            (5.0, 513.6, 513.8, 17718, 17809, 0.0249, 0.0378, 0.0291, 0.0218, 0.0462, 0.0314),
            (4.0, 526.0, 526.3, 18335, 18425, 0.0262, 0.0458, 0.0316, 0.0186, 0.0519, 0.0309),
        )
    ),
}

#: Table 8: how far the model's own delay and area sit from the theoretical
#: ones, in per cent. The geometry route is not expected to beat these.
LITRICO_MAX_RELATIVE_ERRORS = {
    ("wm", "low"): {"delay": 0.61, "area": 3.26},
    ("wm", "high"): {"delay": 0.26, "area": 4.03},
    ("corning", "low"): {"delay": 0.06, "area": 1.13},
    ("corning", "high"): {"delay": 0.31, "area": 2.56},
}


# ---------------------------------------------------------------------------
# Steady flow at a section
# ---------------------------------------------------------------------------


def friction_slope(pool: TrapezoidalPool, depth: float, discharge: float) -> float:
    """Manning-Strickler friction slope, Eq. (33)."""
    area = flow_area(pool, depth)
    radius = hydraulic_radius(pool, depth)
    return discharge**2 * pool.manning_n**2 / (area**2 * radius ** (4.0 / 3.0))


def froude_number(pool: TrapezoidalPool, depth: float, discharge: float) -> float:
    """``V_0 / C_0`` at a section carrying *discharge* at *depth*."""
    return (discharge / flow_area(pool, depth)) / wave_celerity(pool, depth)


def depth_gradient(pool: TrapezoidalPool, depth: float, discharge: float) -> float:
    """``dY_0/dx``, Eq. (32), positive where the pool is backed up.

    Zero at normal depth, which is what makes Eq. (40) treat a pool in
    uniform flow as having no backed-up part at all.
    """
    froude = froude_number(pool, depth, discharge)
    if froude >= 1.0:
        raise ValueError(
            f"{pool.name}: the flow is not subcritical at depth {depth} m "
            f"carrying {discharge} m^3/s, and this model does not apply"
        )
    return (pool.bed_slope - friction_slope(pool, depth, discharge)) / (
        1.0 - froude**2
    )


# ---------------------------------------------------------------------------
# The backwater curve, in two straight pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackwaterSplit:
    """Where a pool stops running uniform and starts being backed up.

    ``uniform_length_m`` is ``x_1`` of Eq. (40): the reach from the head of
    the pool to the point where the tangent to the backwater curve at the
    downstream end meets normal depth. Everything past it is the backed-up
    part, and the two are modelled separately and then joined.
    """

    uniform_length_m: float
    backwater_length_m: float
    normal_depth_m: float
    upstream_depth_m: float
    midpoint_depth_m: float
    surface_slope: float


def backwater_split(
    pool: TrapezoidalPool,
    downstream_depth: float,
    discharge: float,
    profile_slope: str = "surface",
) -> BackwaterSplit:
    """Split the pool into its uniform and its backed-up part, Eqs (39)-(42)."""
    if profile_slope not in PROFILE_SLOPE_READINGS:
        raise ValueError(
            f"profile_slope must be one of {PROFILE_SLOPE_READINGS}, "
            f"got {profile_slope!r}"
        )
    length = pool.length_m
    uniform = normal_depth(pool, discharge)
    slope_at_end = depth_gradient(pool, downstream_depth, discharge)

    if abs(slope_at_end) < 1.0e-15:
        uniform_length = length
    else:
        uniform_length = max(length - (downstream_depth - uniform) / slope_at_end, 0.0)
    uniform_length = min(uniform_length, length)

    slope = slope_at_end if profile_slope == "surface" else pool.bed_slope
    if uniform_length > 0.0:
        upstream_depth = uniform
    else:
        upstream_depth = downstream_depth - length * slope

    midpoint = (uniform_length + length) / 2.0
    return BackwaterSplit(
        uniform_length_m=uniform_length,
        backwater_length_m=length - uniform_length,
        normal_depth_m=uniform,
        upstream_depth_m=upstream_depth,
        midpoint_depth_m=upstream_depth + (midpoint - uniform_length) * slope,
        surface_slope=slope,
    )


# ---------------------------------------------------------------------------
# One homogeneous part, as a two-port
# ---------------------------------------------------------------------------


def _gamma(
    pool: TrapezoidalPool, depth: float, discharge: float, gradient: float
) -> float:
    """``gamma_0`` of Eq. (34), the coefficient that damps the wave."""
    width = top_width(pool, depth)
    area = flow_area(pool, depth)
    velocity = discharge / area
    froude = velocity / wave_celerity(pool, depth)
    kappa = shape_exponent(pool, depth)
    width_gradient = 2.0 * pool.side_slope * gradient
    return velocity**2 * width_gradient + GRAVITY * width * (
        (1.0 + kappa) * pool.bed_slope
        - (1.0 + kappa - (kappa - 2.0) * froude**2) * gradient
    )


def _part_areas(
    pool: TrapezoidalPool,
    depth: float,
    discharge: float,
    gradient: float,
    length: float,
) -> tuple[float, float]:
    """``(A_d, A_u)`` for one homogeneous part, Eqs (49) and (50).

    Both tend to the water-surface area ``T_0 x`` when the exponent is
    small, which is the reading the source gives in words just after
    Eq. (12) and which ``test_a_short_part_stores_its_surface_area``
    checks.
    """
    width = top_width(pool, depth)
    area = flow_area(pool, depth)
    velocity = discharge / area
    celerity = wave_celerity(pool, depth)
    wave = celerity**2 - velocity**2
    gamma = _gamma(pool, depth, discharge, gradient)
    if abs(gamma) < 1.0e-300:
        return width * length, width * length
    scale = width**2 * wave / gamma
    exponent = gamma * length / (width * wave)
    return scale * (1.0 - math.exp(-exponent)), scale * (math.exp(exponent) - 1.0)


def _surface_decay(
    pool: TrapezoidalPool, depth: float, discharge: float, surface_slope: float
) -> float:
    """``alpha`` of Eq. (65), or ``alpha bar`` of Eq. (66) when backed up.

    Equation (66) reduces to Eq. (65) when the surface slope is zero, which
    is the consistency the two printed forms have to satisfy and the
    reading that fixes how Eq. (66)'s brackets nest.

    For uniform flow this equals the round-trip decay rate of
    :func:`faircanal.geometry.wave_decay_rate`, which is written down in a
    different paper from different starting equations - see
    ``test_the_two_papers_agree_on_the_decay_rate``.
    """
    width = top_width(pool, depth)
    area = flow_area(pool, depth)
    froude = froude_number(pool, depth, discharge)
    kappa = shape_exponent(pool, depth)
    lead = width / (area * froude * (1.0 - froude**2))
    term = (2.0 + (kappa - 1.0) * froude**2) * pool.bed_slope
    if surface_slope != 0.0:
        width_gradient = 2.0 * pool.side_slope
        term -= (
            2.0
            + (kappa - 1.0) * froude**2
            - (area / width**2 * width_gradient + kappa - 2.0) * froude**4
        ) * surface_slope
    return lead * term


def _part_gains(
    pool: TrapezoidalPool,
    depth: float,
    discharge: float,
    gradient: float,
    length: float,
) -> tuple[float, float, float, float]:
    """``(p11, p12, p21, p22)`` at infinity for one part, Eqs (61)-(64).

    The reading of Eq. (64) is confirmed by its own limits: at zero length
    it is ``1/(T_0 C_0)`` and at infinite length it is ``1/(T_0 (C_0 -
    V_0))``, and the second of those is exactly Eq. (3) of Clemmens et al.,
    which that paper says it obtained by "modifying their equation for p22
    for a pool of infinite length".
    """
    width = top_width(pool, depth)
    area = flow_area(pool, depth)
    velocity = discharge / area
    celerity = wave_celerity(pool, depth)
    froude = velocity / celerity
    decay = math.exp(_surface_decay(pool, depth, discharge, gradient) * length)
    gamma = _gamma(pool, depth, discharge, gradient)
    half = gamma * length / (2.0 * width * (celerity**2 - velocity**2))

    p11 = (
        1.0
        / (width * celerity * (1.0 - froude))
        * math.sqrt(
            (1.0 + ((1.0 - froude) / (1.0 + froude)) ** 2 * decay) / (1.0 + decay)
        )
    )
    p22 = (
        1.0
        / (width * celerity * (1.0 + froude))
        * math.sqrt(
            (1.0 + ((1.0 + froude) / (1.0 - froude)) ** 2 * decay) / (1.0 + decay)
        )
    )
    common = 2.0 / (width * celerity * (1.0 - froude**2)) / math.sqrt(1.0 + decay)
    p12 = common * math.exp(-half)
    p21 = common * math.exp(half)
    return p11, p12, p21, p22


# ---------------------------------------------------------------------------
# The whole pool
# ---------------------------------------------------------------------------


def idz_from_geometry(
    pool: TrapezoidalPool,
    discharge: float,
    downstream_depth: float | None = None,
    profile_slope: str = "surface",
) -> IdzParams:
    """The four IDZ parameters of one pool, from its geometry alone.

    Parameters
    ----------
    pool:
        The reach.
    discharge:
        Steady discharge through it, m^3/s.
    downstream_depth:
        Depth held at the downstream end, m. Defaults to the pool's
        ``target_level_m``, which is what an upstream controller holds.
    profile_slope:
        Which reading of Eq. (41) to use. See the module docstring; the
        default is the one the published tables select.

    The pool is split by Eq. (40), each part is turned into a two-port by
    Eqs (49)-(50) and (61)-(64), and the two are joined by Eqs (47)-(48)
    and (57)-(60). A part of zero length is dropped rather than joined,
    because the joining equations are not transparent in that limit: they
    were written for two parts that both exist.
    """
    if downstream_depth is None:
        downstream_depth = pool.target_level_m
    if downstream_depth is None:
        raise ValueError(
            f"{pool.name}: no downstream depth was given and the pool has no "
            f"target level, so there is nothing to hold it at"
        )
    if discharge <= 0.0:
        raise ValueError(f"{pool.name}: the discharge must be positive")

    split = backwater_split(pool, downstream_depth, discharge, profile_slope)
    uniform_length = split.uniform_length_m
    backed_length = split.backwater_length_m

    delay = 0.0
    if uniform_length > 0.0:
        depth = split.upstream_depth_m
        speed = wave_celerity(pool, depth) + discharge / flow_area(pool, depth)
        delay += uniform_length / speed
    if backed_length > 0.0:
        depth = split.midpoint_depth_m
        speed = wave_celerity(pool, depth) + discharge / flow_area(pool, depth)
        delay += backed_length / speed

    if uniform_length > 0.0 and backed_length > 0.0:
        area_down, _ = _part_areas(
            pool, split.upstream_depth_m, discharge, 0.0, uniform_length
        )
        backed_down, backed_up = _part_areas(
            pool, split.midpoint_depth_m, discharge, split.surface_slope, backed_length
        )
        area = backed_down * (1.0 + area_down / backed_up)

        _, _, p21, p22 = _part_gains(
            pool, split.upstream_depth_m, discharge, 0.0, uniform_length
        )
        b11, b12, b21, b22 = _part_gains(
            pool, split.midpoint_depth_m, discharge, split.surface_slope, backed_length
        )
        joint = b11 + p22
        gain_21 = p21 * b21 / joint
        gain_22 = b22 + b12 * b21 / joint
    elif backed_length > 0.0:
        area, _ = _part_areas(
            pool, split.midpoint_depth_m, discharge, split.surface_slope, backed_length
        )
        _, _, gain_21, gain_22 = _part_gains(
            pool, split.midpoint_depth_m, discharge, split.surface_slope, backed_length
        )
    else:
        area, _ = _part_areas(
            pool, split.upstream_depth_m, discharge, 0.0, uniform_length
        )
        _, _, gain_21, gain_22 = _part_gains(
            pool, split.upstream_depth_m, discharge, 0.0, uniform_length
        )

    return IdzParams(
        name=f"{pool.name}-idz",
        discharge_m3_s=float(discharge),
        depth_m=float(downstream_depth),
        delay_s=float(delay),
        area_m2=float(area),
        p21=float(gain_21),
        p22=float(gain_22),
        provenance="derived",
        source=LITRICO_SOURCE + " Appendices I-IV, applied to the pool geometry.",
    )


# ---------------------------------------------------------------------------
# How much of a wave comes back
# ---------------------------------------------------------------------------


def round_trip_decay(
    pool: TrapezoidalPool,
    discharge: float,
    downstream_depth: float | None = None,
) -> float:
    """What a gravity wave keeps after running the pool and returning.

    Equation (65) gives the rate at which the wave loses height per metre
    of uniform flow, and Eq. (66) gives it for the backed-up part. The pool
    is both, in the proportions Eq. (40) sets, so the round trip keeps

        exp(-(alpha x1 + alpha_bar (X - x1))).

    Why not the uniform-flow rate for the whole pool
    ------------------------------------------------
    Because a backed-up reach barely damps a wave at all, and using the
    uniform rate there gets the physics backwards. Equation (66) collapses
    to almost nothing when the surface is flat, which is what being backed
    up means; the uniform rate instead grows as the velocity falls, so it
    would say a pool damps waves *better* the less water is moving through
    it. Clemmens et al. say the opposite in words - "the resonance peak at
    low discharges can be significantly higher than that predicted by
    Eq. (3)", and their Eqs (24)-(31) are given under the heading "Uniform
    Flow" - and this reading reproduces that, while the uniform one
    reverses it. Two further checks agree with it and are kept as tests:
    the steep canal's pools 2 to 8 come out with no resonance at all, which
    is what the paper reports of that canal, and the decay computed here
    lands close to the decay implied by the published spread of ``p22``.
    """
    if downstream_depth is None:
        downstream_depth = pool.target_level_m
    if downstream_depth is None:
        raise ValueError(f"{pool.name}: no downstream depth and no target level")
    split = backwater_split(pool, downstream_depth, discharge)
    exponent = 0.0
    if split.uniform_length_m > 0.0:
        exponent += (
            _surface_decay(pool, split.upstream_depth_m, discharge, 0.0)
            * split.uniform_length_m
        )
    if split.backwater_length_m > 0.0:
        exponent += (
            _surface_decay(
                pool, split.midpoint_depth_m, discharge, split.surface_slope
            )
            * split.backwater_length_m
        )
    if exponent < 0.0:
        raise ValueError(
            f"{pool.name}: the wave gains height over a round trip, which means "
            f"the decay rate came out negative ({exponent:.6g}); the operating "
            f"point is outside what this model covers"
        )
    return math.exp(-exponent)


def reflection_amplification(
    pool: TrapezoidalPool,
    discharge: float,
    downstream_depth: float | None = None,
) -> float:
    """How much the reflected waves raise the resonance, as a ratio.

    Clemmens et al.'s Eq. (31) over their Eq. (3), with the round-trip
    decay of :func:`round_trip_decay` in place of their uniform-flow one:

        (1 + (c+v)/(c-v) d) / (1 - d).

    One when nothing comes back, and unbounded as the damping vanishes.
    """
    if downstream_depth is None:
        downstream_depth = pool.target_level_m
    if downstream_depth is None:
        raise ValueError(f"{pool.name}: no downstream depth and no target level")
    decay = round_trip_decay(pool, discharge, downstream_depth)
    if decay >= 1.0:
        raise ValueError(f"{pool.name}: an undamped pool has no finite peak")
    celerity = wave_celerity(pool, downstream_depth)
    velocity = discharge / flow_area(pool, downstream_depth)
    ratio = (celerity + velocity) / (celerity - velocity)
    return (1.0 + ratio * decay) / (1.0 - decay)


def decay_implied_by_published_range(
    pool: TrapezoidalPool, row: LitricoRow, end: str = "downstream"
) -> tuple[float, float]:
    """Read the round-trip decay back out of a published gain range.

    Tables 4 to 7 print the smallest and largest value each high-frequency
    gain takes, and those are the trough and the crest of the same comb:

        max = R (1 + q d) / (1 - d),   min = R (1 - q d) / (1 + d)

    with ``R`` the reflection-free asymptote of Eq. (3) and ``q =
    (c+v)/(c-v)``. Each of the two can be solved for ``d`` on its own, so
    the pair is returned rather than an average - if they disagree, the
    reading is wrong, and that is worth seeing rather than hiding in a
    mean.

    This is the only route to the damping that uses no geometry beyond the
    section itself and no formula from the appendices, which is what makes
    it worth having next to :func:`round_trip_decay`.
    """
    depth = pool.target_level_m
    if depth is None:
        raise ValueError(f"{pool.name}: no target level to evaluate the range at")
    celerity = wave_celerity(pool, depth)
    velocity = row.discharge_m3_s / flow_area(pool, depth)
    ratio = (celerity + velocity) / (celerity - velocity)
    base = 1.0 / (top_width(pool, depth) * (celerity - velocity))

    low, high = (row.p22_range if end == "downstream" else row.p21_range)
    from_crest = (high / base - 1.0) / (high / base + ratio)
    from_trough = (1.0 - low / base) / (ratio + low / base)
    return from_crest, from_trough
