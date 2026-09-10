"""From geometry to a discrete pool model that still carries its wave.

What this module is for
-----------------------
:mod:`faircanal.idz` produces an integrator, a delay and two feedthrough
gains. That is a complete model of a canal pool except for one thing, and
it is the one thing this study cannot do without: the source of the IDZ
model says so itself, that the oscillating modes are left out because
"these modes are usually filtered, in order not to destabilize the
controller". The filter is what this study is about. A plant with no wave
in it would leave the filter unmotivated, and the delivery delay the filter
costs would look like a choice rather than a consequence.

So the first mode is put back, and the model becomes the third-order one
the control study uses:

    y[k+1] = b1 u[k-t] - b2 u[k-t-1] + b3 u[k-t-2]
             - c1 d[k] + c2 d[k-1] - c3 d[k-2]
             + y[k] + a1 (y[k] - 2 y[k-1] + y[k-2]) + a2 (y[k] - y[k-1])

How it is built
---------------
Collected as a transfer function, the level responds to the inflow through

    z^-(t+1) (b1 - b2 z^-1 + b3 z^-2)
    ----------------------------------------------
    (1 - z^-1) (1 - (a1 + a2) z^-1 + a1 z^-2)

and the construction factors that into two pieces that are built
separately and then multiplied:

    IDZ                        resonance
    z^-(t+1) (m1 - m2 z^-1)    1 - a2 z^-1
    ----------------------  x  -------------------------
    1 - z^-1                   1 - (a1 + a2) z^-1 + a1 z^-2

The first is the integrator-delay-zero model written as a difference
equation: ``m1`` is the feedthrough ``p21`` or ``p22`` and ``m1 - m2`` is
the integrator gain ``dt / A_d``. The second has gain exactly one at zero
frequency and leading coefficient exactly one, so it leaves both of those
untouched and does its work in between. Multiplying them out gives

    b1 = m1,  b2 = (1 + a2) m1 - g,  b3 = a2 (m1 - g),   g = dt / A_d

and the same for ``c`` with ``p22`` in place of ``p21``. Nothing is fitted
and nothing is chosen: given the pole pair, the numerator follows.

That second factor is also the only one available. A degree-one factor
that has to be one at ``z = 1`` and one at ``z = infinity`` has no freedom
left - its zero must sit at ``a2`` - so the third-order model cannot hold
both a clean integrator-delay-zero and a clean resonance. What it costs is
a broad rise above the resonance: on the pools of the flat canal that
actually ring, the model's own peak sits within a factor of two of ``w_r``
and within a quarter of the height asked for, while on pools that barely
ring at all the rise wanders out to several times ``w_r``. That is a
property of the model order the source study uses, not of this
construction; ``test_the_peak_lands_near_the_resonance`` measures it and
holds it to what is reported here.

Why the pole pair is not the round-trip decay
---------------------------------------------
A wave that runs the pool and comes back is multiplied by ``exp(-r L)``,
and the obvious move is to give the discrete pole pair a radius that
decays by that much over the ``N = T_cycle / dt`` samples of a round trip.
That is wrong, and wrong by a lot. The real pool is a comb: it resonates at
every multiple of the round-trip frequency, and it has ``N`` pole pairs on
a circle of that radius, not one. At the first tooth of the comb the other
``N - 2`` pairs contribute a product that all but cancels the sharpness of
the nearest one. Keeping one pair and dropping the rest turns a pool whose
peak stands a fifth of a per cent above its asymptote into a model with a
resonance forty times as tall, and no numerator can pull it back down -
``test_the_round_trip_radius_would_not_work`` keeps that from being
forgotten.

So the radius is set by what the reduced model has to reproduce, which is
the height of the peak. Clemmens et al. give it: Eq. (31) over Eq. (3) is
the factor by which reflected waves raise the resonance above the
asymptote, and :func:`pole_radius_for` inverts the shaping factor to find
the radius that produces it. A pool with no reflections gets radius zero
and the model collapses back to the IDZ exactly, which is the right
behaviour and is checked.

How well the damping is known
-----------------------------
The amplification comes from :func:`faircanal.idz.reflection_amplification`,
which uses the two-part decay rather than the uniform-flow one for the
reason given there. It can be checked against the published tables without
using any of that machinery, because Tables 4 to 7 print the smallest and
largest value each high-frequency gain takes, and those are the trough and
the crest of the same comb - see
:func:`faircanal.idz.decay_implied_by_published_range`. On the flat canal
the two routes agree in shape and the geometry route retains rather more of
the wave: about five to ten hundredths more of it per round trip, in the
same direction at every pool and every flow. The gap is reported rather
than tuned away, and the experiments are meant to be run both ways.

What cannot be modelled is refused
----------------------------------
A pool whose resonance is faster than twice the sample time has no
discrete pole pair to carry it. The first reach of the steep test canal is
one hundred metres long and rings with a period of about ninety seconds,
which at a one-minute sample time is below Nyquist. Rather than fold that
wave to some other frequency, :func:`identify_pool` refuses, and the caller
either samples faster or asks for a model without the wave in so many
words.

Sources
-------
Litrico X, Fromion V (2004), doi:10.1061/(ASCE)0733-9437(2004)130:5(373):
the IDZ model, Eqs (3), (4), (43)-(46).

Clemmens AJ, Tian X, van Overloop P-J, Litrico X (2015),
doi:10.1061/(ASCE)IR.1943-4774.0000997: the resonance frequency Eq. (15),
the peak height Eqs (3) and (31), and the decay rate Eqs (25)-(29).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from faircanal.config import DT_PLANT_S
from faircanal.geometry import (
    TrapezoidalPool,
    flow_area,
    resonance_frequency,
)
from faircanal.idz import (
    IdzParams,
    idz_from_geometry,
    reflection_amplification,
    round_trip_decay,
)
from faircanal.pool import PoolParams

__all__ = [
    "IdentificationError",
    "Resonance",
    "resonance_of",
    "shaping_gain",
    "pole_radius_for",
    "pole_coefficients",
    "numerator_for",
    "third_order_from_idz",
    "identify_pool",
    "identify_canal",
    "MAX_POLE_RADIUS",
]

#: A pole this close to the unit circle is a canal that never stops
#: ringing. Nothing published for these canals comes near it; the bound
#: exists so that a bad operating point fails loudly instead of producing a
#: plant that cannot be controlled.
MAX_POLE_RADIUS = 0.999


class IdentificationError(ValueError):
    """Raised when a pool cannot be given a third-order model."""


@dataclass(frozen=True)
class Resonance:
    """The first resonance of a pool, as the pole pair needs to see it.

    ``round_trip_decay`` is ``exp(-r L)`` and ``amplification`` is the
    height of the peak over the asymptote. Both are reported because they
    are the two ways the literature talks about the same wave, and because
    a pool with ``amplification`` near one has, for this study's purposes,
    no wave at all - which is worth knowing before the results are read.
    """

    frequency_rad_s: float
    period_s: float
    samples_per_period: float
    round_trip_decay: float
    amplification: float
    pole_radius: float

    @property
    def angle_rad(self) -> float:
        """Pole angle, ``w_r dt``, in radians per sample."""
        return 2.0 * math.pi / self.samples_per_period

    @property
    def resolvable(self) -> bool:
        """Whether the wave is above the Nyquist limit of the sample time."""
        return self.samples_per_period > 2.0


def shaping_gain(radius: float, angle: float) -> float:
    """Height of the resonance a pole pair produces, as a ratio.

    The shaping factor ``(1 - a2 z^-1) / (1 - (a1 + a2) z^-1 + a1 z^-2)``
    is one at zero frequency and one as ``z`` grows, so its magnitude at
    the resonance is the amplification the model adds there. At radius zero
    it is one everywhere - no wave - and it grows without bound as the
    radius approaches the unit circle, which is what makes it invertible.
    """
    if not 0.0 <= radius < 1.0:
        raise ValueError(f"the pole radius must be in [0, 1), got {radius}")
    a1 = radius**2
    a2 = 2.0 * radius * math.cos(angle) - a1
    unit = complex(math.cos(-angle), math.sin(-angle))
    numerator = abs(1.0 - a2 * unit)
    denominator = abs(1.0 - (a1 + a2) * unit + a1 * unit**2)
    return numerator / denominator


def pole_radius_for(amplification: float, angle: float) -> float:
    """The pole radius whose resonance stands *amplification* times high.

    Inverts :func:`shaping_gain`, which rises from one at radius zero to
    infinity at the unit circle. An amplification of one means no
    reflections come back and the answer is zero, which makes the whole
    resonance factor vanish and leaves the IDZ model untouched.
    """
    if amplification < 1.0:
        raise IdentificationError(
            f"an amplification of {amplification:.6g} is below one, so the "
            f"reflected waves would have to make the peak smaller; nothing in "
            f"the source's Eq. (31) can do that"
        )
    if amplification == 1.0:
        return 0.0
    if shaping_gain(MAX_POLE_RADIUS, angle) < amplification:
        raise IdentificationError(
            f"an amplification of {amplification:.6g} needs a pole closer to the "
            f"unit circle than {MAX_POLE_RADIUS}; this pool rings far harder than "
            f"anything published for these canals"
        )
    return float(
        brentq(
            lambda radius: shaping_gain(radius, angle) - amplification,
            0.0,
            MAX_POLE_RADIUS,
            xtol=1e-14,
            rtol=8.9e-16,
        )
    )


def resonance_of(
    pool: TrapezoidalPool,
    discharge: float,
    downstream_depth: float | None = None,
    dt: float = DT_PLANT_S,
) -> Resonance:
    """Everything about the first resonance that the model needs."""
    depth = pool.target_level_m if downstream_depth is None else downstream_depth
    if depth is None:
        raise IdentificationError(
            f"{pool.name}: no downstream depth was given and the pool has no "
            f"target level"
        )
    velocity = discharge / flow_area(pool, depth)
    omega = resonance_frequency(pool, depth, velocity)
    period = 2.0 * math.pi / omega
    samples = period / dt
    amplification = reflection_amplification(pool, discharge, depth)
    radius = (
        pole_radius_for(amplification, 2.0 * math.pi / samples)
        if samples > 2.0
        else float("nan")
    )
    return Resonance(
        frequency_rad_s=omega,
        period_s=period,
        samples_per_period=samples,
        round_trip_decay=round_trip_decay(pool, discharge, depth),
        amplification=amplification,
        pole_radius=radius,
    )


def pole_coefficients(resonance: Resonance) -> tuple[float, float]:
    """``(a1, a2)`` from the pole radius and angle.

    The homogeneous part of the model has characteristic polynomial
    ``(z - 1)(z^2 - (a1 + a2) z + a1)``, so a pole pair of radius ``rho``
    and angle ``theta`` needs ``a1 = rho^2`` and ``a1 + a2 = 2 rho cos
    theta``. The unit root is the integrator and is not ours to choose.
    """
    if not resonance.resolvable:
        raise IdentificationError(
            f"the first resonance has a period of {resonance.period_s:.1f} s, "
            f"which is {resonance.samples_per_period:.2f} samples. A wave below "
            f"two samples per period has no discrete pole pair."
        )
    radius = resonance.pole_radius
    angle = resonance.angle_rad
    a1 = radius**2
    a2 = 2.0 * radius * math.cos(angle) - a1
    return a1, a2


def numerator_for(
    feedthrough: float, integrator_gain: float, a2: float
) -> tuple[float, float, float]:
    """The three numerator coefficients of one channel.

    Multiplying out the IDZ numerator ``m1 - m2 z^-1`` and the resonance
    numerator ``1 - a2 z^-1``, with ``m1`` the feedthrough and ``m1 - m2``
    the integrator gain. The signs follow the model's own convention, in
    which the middle coefficient is subtracted.
    """
    m1 = feedthrough
    return (
        m1,
        (1.0 + a2) * m1 - integrator_gain,
        a2 * (m1 - integrator_gain),
    )


def third_order_from_idz(
    name: str,
    idz: IdzParams,
    resonance: Resonance,
    dt: float = DT_PLANT_S,
    source: str = "",
) -> PoolParams:
    """Assemble the discrete third-order model of one pool."""
    if not resonance.resolvable:
        raise IdentificationError(
            f"{name}: the first resonance has a period of "
            f"{resonance.period_s:.1f} s, which is "
            f"{resonance.samples_per_period:.2f} samples at dt = {dt:g} s. A wave "
            f"below two samples per period has no discrete pole pair, so this pool "
            f"cannot carry its wave at this sample time."
        )
    a1, a2 = pole_coefficients(resonance)
    gain = dt / idz.area_m2
    return PoolParams(
        name=name,
        b=numerator_for(idz.p21, gain, a2),
        c=numerator_for(idz.p22, gain, a2),
        alpha=(a1, a2),
        tau=int(round(idz.delay_s / dt)),
        units="s/m^2",
        provenance="derived",
        source=source or idz.source,
    )


def identify_pool(
    pool: TrapezoidalPool,
    discharge: float,
    downstream_depth: float | None = None,
    dt: float = DT_PLANT_S,
) -> PoolParams:
    """Identify one pool at one operating point, from geometry alone."""
    idz = idz_from_geometry(pool, discharge, downstream_depth)
    return third_order_from_idz(
        f"{pool.name}-order-3",
        idz,
        resonance_of(pool, discharge, downstream_depth, dt),
        dt,
        source=(
            f"{idz.source} Resonance and reflection from Clemmens AJ, Tian X, "
            f"van Overloop P-J, Litrico X (2015), "
            f"doi:10.1061/(ASCE)IR.1943-4774.0000997, Eqs (3), (15), (25)-(31). "
            f"Operating point: {discharge:g} m^3/s."
        ),
    )


def identify_canal(
    pools: "tuple[TrapezoidalPool, ...] | list[TrapezoidalPool]",
    discharges: "tuple[float, ...] | list[float]",
    dt: float = DT_PLANT_S,
) -> tuple[PoolParams, ...]:
    """Identify a whole canal, one operating point per pool.

    ``discharges[i]`` is the steady discharge through pool ``i``, most
    upstream first, which is how the published tables of both test canals
    are laid out.
    """
    if len(pools) != len(discharges):
        raise IdentificationError(f"{len(pools)} pools but {len(discharges)} discharges")
    return tuple(
        identify_pool(pool, discharge, dt=dt)
        for pool, discharge in zip(pools, discharges)
    )
