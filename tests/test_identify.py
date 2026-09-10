"""Permanent tests for the pool identification. Never deleted.

The plant of every experiment comes out of this module, so an error here
would not break anything - it would produce a canal that behaves plausibly
and is not the ASCE canal. The tests are therefore built around statements
made outside this code: a published sentence about which pools oscillate, a
published sentence about what low flow does to the peak, and the arithmetic
identities the construction is supposed to preserve exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from faircanal.config import DT_PLANT_S
from faircanal.geometry import CORNING_POOLS, WM_POOLS
from faircanal.identify import (
    IdentificationError,
    MAX_POLE_RADIUS,
    identify_canal,
    identify_pool,
    numerator_for,
    pole_coefficients,
    pole_radius_for,
    resonance_of,
    shaping_gain,
    third_order_from_idz,
)
from faircanal.idz import LITRICO_TABLES, idz_from_geometry
from faircanal.pool import homogeneous_roots, ramp_rate, simulate_level

CORNING_HIGH = tuple(row.discharge_m3_s for row in LITRICO_TABLES[("corning", "high")])
CORNING_LOW = tuple(row.discharge_m3_s for row in LITRICO_TABLES[("corning", "low")])


def corning_models(flow: str = "high"):
    """Every Corning pool at one published operating point."""
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    return identify_canal(CORNING_POOLS, discharges)


def response_at(params, angle: float, channel: str = "c") -> float:
    """Magnitude of the model's transfer function at one frequency.

    The delay and the leading ``z^-1`` have unit magnitude, so what is left
    is the numerator over the denominator.
    """
    a1, a2 = params.alpha
    unit = complex(math.cos(-angle), math.sin(-angle))
    coefficients = params.c if channel == "c" else params.b
    n1, n2, n3 = coefficients
    numerator = abs(n1 - n2 * unit + n3 * unit**2)
    denominator = abs((1.0 - unit) * (1.0 - (a1 + a2) * unit + a1 * unit**2))
    return numerator / denominator


def idz_magnitude(idz, omega: float, channel: str = "c") -> float:
    """What the continuous IDZ model says at that frequency."""
    feedthrough = idz.p22 if channel == "c" else idz.p21
    return math.hypot(feedthrough, 1.0 / (idz.area_m2 * omega))


def discrete_idz_magnitude(idz, angle: float, channel: str = "c") -> float:
    """The same model as a difference equation, before the wave is added.

    ``(m1 - m2 z^-1) / (1 - z^-1)`` with ``m1`` the feedthrough and
    ``m1 - m2`` the integrator gain. This, and not its continuous parent,
    is what the resonance factor multiplies.
    """
    feedthrough = idz.p22 if channel == "c" else idz.p21
    gain = DT_PLANT_S / idz.area_m2
    unit = complex(math.cos(-angle), math.sin(-angle))
    return abs(feedthrough - (feedthrough - gain) * unit) / abs(1.0 - unit)


# ---------------------------------------------------------------------------
# What the construction promises to preserve exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_integrator_gain_survives_the_resonance(flow: str):
    """A sustained flow ramps the level at the rate the backwater area says.

    This is the one property that has to be exact rather than close: it is
    mass conservation. If the resonance factor moved it, water would appear
    or vanish in proportion to how much the pool rings, and every delivered
    volume in the study would be wrong by a pool-dependent factor.
    """
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    for pool, discharge in zip(CORNING_POOLS, discharges):
        model = identify_pool(pool, discharge)
        idz = idz_from_geometry(pool, discharge)
        expected = DT_PLANT_S / idz.area_m2
        inflow, outflow = ramp_rate(model)
        assert inflow == pytest.approx(expected, rel=1e-12), pool.name
        assert outflow == pytest.approx(expected, rel=1e-12), pool.name


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_feedthrough_survives_the_resonance(flow: str):
    """The instantaneous response is the IDZ gain, unchanged.

    The resonance factor has leading coefficient one for exactly this
    reason: a gate movement is felt at the sensor beside it at once, and by
    the amount the high-frequency gain says.
    """
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    for pool, discharge in zip(CORNING_POOLS, discharges):
        model = identify_pool(pool, discharge)
        idz = idz_from_geometry(pool, discharge)
        assert model.b[0] == pytest.approx(idz.p21, rel=1e-15)
        assert model.c[0] == pytest.approx(idz.p22, rel=1e-15)


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_resonance_stands_as_high_as_asked(flow: str):
    """At the resonance the model is exactly the amplification above the IDZ.

    That is the whole content of the pole radius: the frequency comes from
    the round trip and the height comes from the reflections, and this
    checks that inverting the shaping factor really put it there.
    """
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    for pool, discharge in zip(CORNING_POOLS, discharges):
        model = identify_pool(pool, discharge)
        idz = idz_from_geometry(pool, discharge)
        resonance = resonance_of(pool, discharge)
        for channel in ("b", "c"):
            got = response_at(model, resonance.angle_rad, channel)
            want = resonance.amplification * discrete_idz_magnitude(
                idz, resonance.angle_rad, channel
            )
            assert got == pytest.approx(want, rel=1e-9), f"{pool.name} {channel}"


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_difference_equation_still_is_the_idz_model(flow: str):
    """The discretisation itself, measured at the frequency that matters.

    The resonance is added to the IDZ model written as a difference
    equation, and that difference equation is not quite its continuous
    parent - a step that is exact at zero frequency and at infinity has to
    give a little in between. It gives most where the resonance is fastest
    relative to the sample time, which is the short pools at the tail of
    the canal, and there it gives about five per cent. That is the number
    the manuscript reports for the discretisation.

    The difference equation itself is not in doubt: an impulse of discharge
    drops the level by the feedthrough at once and leaves it lower by the
    volume taken, which is what the continuous model says happens and what
    ``test_an_outflow_is_felt_at_once_and_lowers_the_level`` checks. The
    five per cent is the price of sampling, not a mistake in the mapping.
    """
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    worst = 0.0
    for pool, discharge in zip(CORNING_POOLS, discharges):
        idz = idz_from_geometry(pool, discharge)
        resonance = resonance_of(pool, discharge)
        for channel in ("b", "c"):
            continuous = idz_magnitude(idz, resonance.frequency_rad_s, channel)
            discrete = discrete_idz_magnitude(idz, resonance.angle_rad, channel)
            worst = max(worst, abs(discrete - continuous) / continuous)
    assert worst < 0.06, f"the discretisation now costs {100 * worst:.2f} per cent"


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_model_is_an_integrator_carrying_a_damped_wave(flow: str):
    """One root on the unit circle, two strictly inside, and complex."""
    for model in corning_models(flow):
        roots = homogeneous_roots(model)
        assert len(roots) == 3
        assert abs(roots[0]) == pytest.approx(1.0, abs=1e-12)
        assert all(abs(root) < 1.0 - 1e-9 for root in roots[1:])
        if model.alpha[0] > 1e-12:
            assert abs(roots[1].imag) > 1e-9, f"{model.name}: the wave is not a wave"


@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_transport_delay_is_the_travel_time_in_steps(flow: str):
    """``tau`` is the IDZ delay rounded to whole samples, and no more.

    Rounding is the only thing that happens to it, so the model's delay may
    never sit more than half a sample from the physical one.
    """
    discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
    for pool, discharge in zip(CORNING_POOLS, discharges):
        model = identify_pool(pool, discharge)
        idz = idz_from_geometry(pool, discharge)
        assert abs(model.tau * DT_PLANT_S - idz.delay_s) <= DT_PLANT_S / 2.0 + 1e-9


def test_an_inflow_reaches_the_level_after_the_transport_delay():
    """The delay in the difference equation is a delay in the simulation.

    Driving one pool with a single step of inflow and watching when the
    level first moves is the only way to be sure the index arithmetic in
    the model agrees with the index arithmetic in the simulator.
    """
    pool = CORNING_POOLS[1]
    model = identify_pool(pool, CORNING_HIGH[1])
    steps = 60
    inflow = np.zeros(steps)
    inflow[0] = 1.0
    level = simulate_level(model, inflow, np.zeros(steps), np.zeros(steps))
    moved = int(np.argmax(np.abs(level) > 1e-15))
    assert moved == model.tau + 1
    assert level[moved] == pytest.approx(model.b[0], rel=1e-12)


def test_an_outflow_is_felt_at_once_and_lowers_the_level():
    """No delay at the downstream end, and the sign is the physical one."""
    pool = CORNING_POOLS[1]
    model = identify_pool(pool, CORNING_HIGH[1])
    steps = 10
    drawn = np.zeros(steps)
    drawn[0] = 1.0
    level = simulate_level(model, np.zeros(steps), drawn, np.zeros(steps))
    assert level[1] == pytest.approx(-model.c[0], rel=1e-12)
    assert level[1] < 0.0


# ---------------------------------------------------------------------------
# The pole radius, and the mapping that would have been wrong
# ---------------------------------------------------------------------------


def test_the_shaping_gain_rises_with_the_radius():
    """Monotone from one to unbounded, which is what makes it invertible.

    That holds for every resonance this study meets: the pools of both
    canals sit between about a tenth of a radian and one radian per sample.
    Well past that, close to Nyquist, the gain dips a fraction of a per
    cent below one before it climbs, so the inversion could in principle
    land on either side of a shallow dip. It is checked rather than assumed
    that no pool of this study is anywhere near there.
    """
    for angle in (0.05, 0.25, 0.5, 1.0, 1.5):
        values = [shaping_gain(radius, angle) for radius in np.linspace(0.0, 0.99, 60)]
        assert values[0] == pytest.approx(1.0, abs=1e-12)
        assert all(later > earlier for earlier, later in zip(values, values[1:]))
        assert values[-1] > 10.0

    dip = min(shaping_gain(radius, 2.5) for radius in np.linspace(0.0, 0.5, 40))
    assert dip < 1.0, "the dip near Nyquist has gone, so this caveat can go too"

    for canal, discharges in (
        (CORNING_POOLS, CORNING_LOW),
        (CORNING_POOLS, CORNING_HIGH),
    ):
        for pool, discharge in zip(canal, discharges):
            assert resonance_of(pool, discharge).angle_rad < 1.0


def test_no_reflections_means_no_wave_and_the_idz_comes_back():
    """A pool that reflects nothing gets a model that is exactly the IDZ.

    The whole resonance apparatus has to vanish without a trace in that
    limit, or every weakly reflecting pool would carry a spurious mode.
    """
    assert pole_radius_for(1.0, 0.3) == 0.0
    a1, a2 = 0.0, 0.0
    feedthrough, gain = 0.03, 0.002
    assert numerator_for(feedthrough, gain, a2) == (
        feedthrough,
        feedthrough - gain,
        0.0,
    )


def test_the_round_trip_radius_would_not_work():
    """The mapping that looks obvious, and what it does.

    Giving the single pole pair a radius that decays by ``exp(-r L)`` over
    the samples of a round trip treats one tooth of a comb as if it were
    the whole comb. On the first pool of the flat canal at high flow that
    turns a peak three per cent above the asymptote into one about seven
    times it. The test is here because the mistake is natural, the
    result is plausible-looking, and nothing else in the suite would catch
    it.
    """
    pool = CORNING_POOLS[0]
    resonance = resonance_of(pool, CORNING_HIGH[0])
    naive = resonance.round_trip_decay ** (1.0 / resonance.samples_per_period)
    assert naive > 0.8
    assert shaping_gain(naive, resonance.angle_rad) > 6.0
    assert resonance.amplification < 1.1
    assert resonance.pole_radius < 0.6 * naive


def test_a_wave_faster_than_the_sample_time_is_refused():
    """The steep canal's first reach, which rings in about ninety seconds."""
    for flow in ("low", "high"):
        discharge = LITRICO_TABLES[("wm", flow)][0].discharge_m3_s
        resonance = resonance_of(WM_POOLS[0], discharge)
        assert resonance.samples_per_period < 2.0
        assert not resonance.resolvable
        with pytest.raises(IdentificationError, match="two samples per period"):
            identify_pool(WM_POOLS[0], discharge)
        with pytest.raises(IdentificationError, match="two samples per period"):
            pole_coefficients(resonance)


def test_the_same_reach_can_be_identified_at_a_shorter_sample_time():
    """And that the refusal is about the sample time, not about the reach."""
    model = identify_pool(WM_POOLS[0], 2.0, dt=10.0)
    assert model.order == 3
    assert model.alpha[0] > 0.0


def test_a_pool_that_never_stops_ringing_is_refused():
    with pytest.raises(IdentificationError, match="unit circle"):
        pole_radius_for(1.0e6, 0.2)
    assert shaping_gain(MAX_POLE_RADIUS, 0.2) < 1.0e6


def test_an_amplification_below_one_is_refused():
    with pytest.raises(IdentificationError, match="below one"):
        pole_radius_for(0.9, 0.2)


def test_a_canal_and_its_flows_have_to_match():
    with pytest.raises(IdentificationError, match="pools but"):
        identify_canal(CORNING_POOLS, CORNING_HIGH[:4])


def test_a_pool_without_a_target_level_needs_a_depth():
    from faircanal.geometry import TrapezoidalPool

    pool = TrapezoidalPool(
        name="nameless",
        length_m=2000.0,
        bed_width_m=5.0,
        side_slope=1.5,
        canal_depth_m=2.0,
        manning_n=0.02,
        bed_slope=1.0e-4,
    )
    with pytest.raises(IdentificationError, match="target level"):
        resonance_of(pool, 5.0)
    assert resonance_of(pool, 5.0, downstream_depth=1.7).frequency_rad_s > 0.0


# ---------------------------------------------------------------------------
# What the models say about these canals
# ---------------------------------------------------------------------------


def test_the_peak_lands_near_the_resonance():
    """Where the model's own maximum actually sits, measured.

    The third-order model cannot hold a clean integrator-delay-zero and a
    clean resonance at the same time - the numerator's one free zero is
    spent on the integrator gain - so its maximum is pulled above the
    resonance frequency. On the pools that ring the pull is small; on the
    pools that barely ring it is large but the bump it moves is small too.
    Both are measured here rather than assumed, because the size of this
    artefact is something the manuscript has to state.
    """
    grid = np.linspace(1e-4, math.pi - 1e-4, 4000)
    unit = np.exp(-1j * grid)
    worst_ratio = 0.0
    for flow in ("low", "high"):
        discharges = CORNING_HIGH if flow == "high" else CORNING_LOW
        for pool, discharge in zip(CORNING_POOLS, discharges):
            model = identify_pool(pool, discharge)
            resonance = resonance_of(pool, discharge)
            a1, a2 = model.alpha
            shape = np.abs(1.0 - a2 * unit) / np.abs(
                1.0 - (a1 + a2) * unit + a1 * unit**2
            )
            peak = int(np.argmax(shape))
            where = grid[peak] / resonance.angle_rad
            height = shape[peak] / resonance.amplification
            if resonance.amplification > 2.0:
                assert where < 2.1, f"{pool.name} {flow}: peak at {where:.2f} w_r"
                assert height < 1.3, f"{pool.name} {flow}: peak {height:.2f} x asked"
            worst_ratio = max(worst_ratio, height)
    assert worst_ratio < 1.6, f"the artefact has grown to {worst_ratio:.2f}"


def test_the_flat_canal_rings_harder_at_low_flow():
    """Which operating point the filter is really needed at.

    Held at the same target level, a pool carrying little water is deeply
    backed up and hardly damps a wave at all. The models say so: at the
    published low flow the peaks stand between two and eight times the
    asymptote, and at the published high flow between one and two. That is
    the source's own sentence about low discharges, arriving as a number.
    """
    low = [resonance_of(p, q).amplification for p, q in zip(CORNING_POOLS, CORNING_LOW)]
    high = [
        resonance_of(p, q).amplification for p, q in zip(CORNING_POOLS, CORNING_HIGH)
    ]
    assert all(a > b for a, b in zip(low, high))
    assert 2.0 < min(low) and max(low) < 9.0
    assert 1.0 < min(high) and max(high) < 2.5


def test_every_resonance_sits_where_the_filter_can_see_it():
    """The wave and the filter have to be in the same part of the spectrum.

    The source study's filter has a cut-off of 3e-3 rad/s. Half of this
    canal's pools resonate below that and half above, so the filter as
    published does not cover the canal - which is a finding to report, not
    a bug to hide, and the reason the experiments design their own filter
    for the canal they run on.
    """
    cutoff = 3.0e-3
    frequencies = [
        resonance_of(pool, discharge).frequency_rad_s
        for pool, discharge in zip(CORNING_POOLS, CORNING_HIGH)
    ]
    assert min(frequencies) < cutoff < max(frequencies)
    assert min(frequencies) > 1.0e-3
    assert max(frequencies) < 1.0e-2


def test_a_model_carries_its_own_provenance():
    model = identify_pool(CORNING_POOLS[0], CORNING_HIGH[0])
    assert model.provenance == "derived"
    assert "10.1061/(ASCE)0733-9437(2004)130:5(373)" in model.source
    assert "10.1061/(ASCE)IR.1943-4774.0000997" in model.source
    assert "11 m^3/s" in model.source


def test_the_assembled_model_matches_its_pieces():
    """``third_order_from_idz`` and ``identify_pool`` are the same thing."""
    pool = CORNING_POOLS[5]
    discharge = CORNING_HIGH[5]
    direct = identify_pool(pool, discharge)
    assembled = third_order_from_idz(
        direct.name,
        idz_from_geometry(pool, discharge),
        resonance_of(pool, discharge),
    )
    assert assembled.b == direct.b
    assert assembled.c == direct.c
    assert assembled.alpha == direct.alpha
    assert assembled.tau == direct.tau
