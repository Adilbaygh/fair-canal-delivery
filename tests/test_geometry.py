"""Permanent tests for the canal geometry. Never deleted.

Transcribed tables are the quietest place for an error to live: a wrong
digit produces a canal that is merely a different canal, and every result
downstream is internally consistent and wrong. So the transcriptions are
checked against something other than themselves - a mass balance the
published flows have to satisfy, and worked numbers the sources print.
"""

from __future__ import annotations

import math

import pytest

from faircanal.geometry import (
    CORNING_CHECK_FLOWS,
    CORNING_GATES,
    CORNING_HEADING_FLOW,
    CORNING_OFFTAKES,
    CORNING_POOLS,
    GRAVITY,
    Gate,
    TrapezoidalPool,
    WM_CHECK_FLOWS,
    WM_GATES,
    WM_HEADING_FLOW,
    WM_OFFTAKES,
    WM_POOLS,
    backwater_area,
    flow_area,
    gate_capacity,
    hydraulic_depth,
    normal_depth,
    resonance_frequency,
    resonance_peak_height,
    top_width,
    uniform_discharge,
    wave_celerity,
)

ALL_POOLS = [*CORNING_POOLS, *WM_POOLS]


# ---------------------------------------------------------------------------
# The transcriptions, checked against something other than themselves
# ---------------------------------------------------------------------------


def test_the_corning_flows_balance_reach_by_reach():
    """Every drop entering the canal leaves it through an offtake or a check.

    The published table gives the heading flow, the eight offtakes and the
    eight check flows independently, so they are free to disagree - and
    they do not, at any reach. A single mistyped digit anywhere in the
    seventeen numbers breaks this.
    """
    upstream = CORNING_HEADING_FLOW
    for reach, (offtake, check) in enumerate(
        zip(CORNING_OFFTAKES, CORNING_CHECK_FLOWS), start=1
    ):
        assert upstream - offtake == pytest.approx(check, abs=1e-9), (
            f"reach {reach}: {upstream} - {offtake} is not {check}"
        )
        upstream = check


def test_the_wm_flows_balance_reach_by_reach():
    """The same canal-wide check on the smaller canal."""
    upstream = WM_HEADING_FLOW
    for reach, (offtake, check) in enumerate(
        zip(WM_OFFTAKES, WM_CHECK_FLOWS), start=1
    ):
        assert upstream - offtake == pytest.approx(check, abs=1e-9), (
            f"reach {reach}: {upstream} - {offtake} is not {check}"
        )
        upstream = check
    assert upstream == pytest.approx(0.0, abs=1e-9), "the canal must run dry at its end"


def test_the_worked_example_of_the_source_is_reproduced():
    """Three numbers the source prints, recomputed from the geometry.

    Its response example describes the first reach of the smaller canal at
    a depth of nine tenths of a metre: a top width of 3.7 m, a celerity of
    2.37 m/s, and - at the capacity of that reach - a resonance peak of
    0.190 in its Table 3. All three follow from the bed width, the side
    slope and the depth, so agreeing with them checks the geometry and the
    relations at once.
    """
    pool = WM_POOLS[0]
    depth = 0.9

    assert top_width(pool, depth) == pytest.approx(3.7, abs=5e-3)
    assert wave_celerity(pool, depth) == pytest.approx(2.37, abs=5e-3)

    capacity = 2.0
    velocity = capacity / flow_area(pool, depth)
    assert resonance_peak_height(pool, depth, velocity) == pytest.approx(
        0.190, abs=5e-4
    )


def test_the_canals_have_the_shape_their_sources_describe():
    """Guard the summary facts, which are stated in prose as well as tables."""
    assert len(CORNING_POOLS) == 8 and len(WM_POOLS) == 8
    assert len(CORNING_GATES) == 7 and len(WM_GATES) == 7, (
        "the last reach ends the canal and has no check gate"
    )
    assert sum(p.length_m for p in CORNING_POOLS) == pytest.approx(28_000.0)
    assert sum(p.length_m for p in WM_POOLS) == pytest.approx(9_500.0)
    assert all(p.bed_slope == 1.0e-4 for p in CORNING_POOLS)
    assert all(p.bed_slope == 0.002 for p in WM_POOLS)
    assert all(p.manning_n == 0.02 for p in CORNING_POOLS)
    assert all(p.manning_n == 0.014 for p in WM_POOLS)
    assert all(p.side_slope == 1.5 for p in ALL_POOLS)


def test_corning_is_the_flatter_and_larger_canal():
    """Why it is the primary case, as a property rather than a remark.

    The study is about suppressing waves, and the other canal is reported
    by its own source not to oscillate except in its first reach. Corning
    is flatter, longer and carries an order of magnitude more water, which
    is what puts its reaches under backwater.
    """
    assert CORNING_POOLS[0].bed_slope < WM_POOLS[0].bed_slope / 10.0
    assert sum(p.length_m for p in CORNING_POOLS) > 2.0 * sum(
        p.length_m for p in WM_POOLS
    )
    assert CORNING_HEADING_FLOW > 10.0 * WM_HEADING_FLOW


# ---------------------------------------------------------------------------
# Hydraulics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool", ALL_POOLS, ids=lambda p: p.name)
def test_normal_depth_inverts_the_discharge_relation(pool: TrapezoidalPool):
    for depth in (0.1, 0.5, 1.0):
        discharge = uniform_discharge(pool, depth)
        assert normal_depth(pool, discharge) == pytest.approx(depth, rel=1e-9)


@pytest.mark.parametrize("pool", ALL_POOLS, ids=lambda p: p.name)
def test_discharge_rises_with_depth(pool: TrapezoidalPool):
    """Monotone, which is what makes the inversion's root unique."""
    depths = [0.2, 0.4, 0.8, 1.2, 1.6]
    flows = [uniform_discharge(pool, d) for d in depths]
    assert all(b > a for a, b in zip(flows, flows[1:]))


@pytest.mark.parametrize("pool", ALL_POOLS, ids=lambda p: p.name)
def test_hydraulic_depth_is_below_the_depth_itself(pool: TrapezoidalPool):
    """In a trapezoid the two differ, and the wave speed follows the smaller.

    Using the depth in place of the hydraulic depth would raise every
    celerity and move every resonance, without anything failing.
    """
    for depth in (0.3, 0.9, 1.5):
        assert hydraulic_depth(pool, depth) < depth
        assert wave_celerity(pool, depth) < math.sqrt(GRAVITY * depth)


def test_a_rectangular_channel_has_hydraulic_depth_equal_to_its_depth():
    """The limiting case, where the distinction disappears."""
    rectangle = TrapezoidalPool(
        name="rectangle",
        length_m=100.0,
        bed_width_m=4.0,
        side_slope=0.0,
        canal_depth_m=2.0,
        manning_n=0.02,
        bed_slope=1e-4,
    )
    assert hydraulic_depth(rectangle, 1.3) == pytest.approx(1.3)
    assert top_width(rectangle, 1.3) == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# The resonance
# ---------------------------------------------------------------------------


def test_a_longer_reach_resonates_more_slowly():
    """The wave has further to travel, so the round trip takes longer."""
    short, long = WM_POOLS[0], WM_POOLS[4]
    depth, velocity = 0.9, 0.3
    assert resonance_frequency(long, depth, velocity) < resonance_frequency(
        short, depth, velocity
    )


def test_the_resonance_is_where_a_wave_round_trip_puts_it():
    """Derived rather than restated: period equals the round-trip time."""
    pool = WM_POOLS[0]
    depth, velocity = 0.9, 0.4728
    celerity = wave_celerity(pool, depth)
    cycle = pool.length_m * (1 / (celerity + velocity) + 1 / (celerity - velocity))
    assert resonance_frequency(pool, depth, velocity) == pytest.approx(
        2 * math.pi / cycle, rel=1e-12
    )


def test_supercritical_flow_is_refused_rather_than_returned():
    """A wave that cannot travel upstream has no round trip and no resonance."""
    pool = WM_POOLS[0]
    with pytest.raises(ValueError, match="supercritical"):
        resonance_frequency(pool, 0.9, 5.0)
    with pytest.raises(ValueError, match="supercritical"):
        resonance_peak_height(pool, 0.9, 5.0)


def test_the_backwater_area_is_the_surface_of_the_reach():
    pool = WM_POOLS[0]
    assert backwater_area(pool, 0.9) == pytest.approx(3.7 * 100.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gates, and refusals
# ---------------------------------------------------------------------------


def test_gate_capacity_follows_the_orifice_relation():
    gate = CORNING_GATES[0]
    head = 0.2
    expected = 0.61 * 7.0 * 2.3 * math.sqrt(2 * GRAVITY * head)
    assert gate_capacity(gate, head) == pytest.approx(expected)
    assert gate_capacity(gate, 0.0) == 0.0


def test_gate_capacity_rises_with_the_head_drop():
    gate = CORNING_GATES[0]
    assert gate_capacity(gate, 0.4) > gate_capacity(gate, 0.1)


def test_impossible_geometry_is_refused():
    with pytest.raises(ValueError, match="positive"):
        TrapezoidalPool(
            name="bad", length_m=0.0, bed_width_m=1.0, side_slope=1.5,
            canal_depth_m=1.0, manning_n=0.02, bed_slope=1e-4,
        )
    with pytest.raises(ValueError, match="discharge coefficient"):
        Gate(name="bad", width_m=1.0, height_m=1.0, discharge_coefficient=1.4)
    with pytest.raises(ValueError, match="cannot be negative"):
        gate_capacity(CORNING_GATES[0], -1.0)


@pytest.mark.parametrize("pool", ALL_POOLS, ids=lambda p: p.name)
def test_every_pool_records_its_source(pool: TrapezoidalPool):
    assert pool.provenance in {"observed", "derived", "assumed"}
    assert "doi:" in pool.source


#: Clemmens et al., Table 3: the discharge each reach carries at capacity,
#: and the resonance peak height Eq. (3) gives there.
WM_CAPACITY = (2.0, 2.0, 2.0, 1.6, 1.6, 1.6, 1.3, 1.1)
WM_RESONANCE_PEAK = (0.190, 0.190, 0.264, 0.190, 0.190, 0.257, 0.263, 0.237)


@pytest.mark.parametrize(
    ("index", "capacity", "published"),
    list(zip(range(1, 9), WM_CAPACITY, WM_RESONANCE_PEAK)),
    ids=[f"wm-{i}" for i in range(1, 9)],
)
def test_every_published_resonance_peak_is_reproduced(
    index: int, capacity: float, published: float
):
    """All eight of the source's published peaks, from geometry alone.

    This is the strongest check the transcription can be given. Each value
    uses that reach's bed width, side slope, target level and capacity, so
    reproducing all eight to the three digits the source prints means every
    one of those thirty-two numbers is right. A single wrong digit moves
    its own reach and leaves the others alone, which is exactly what makes
    the check sharp.

    The depth is the target level: the source's own worked example uses it
    for the first reach, and it is what the canal is operated at.
    """
    pool = WM_POOLS[index - 1]
    depth = pool.target_level_m
    velocity = capacity / flow_area(pool, depth)
    assert resonance_peak_height(pool, depth, velocity) == pytest.approx(
        published, abs=1e-3
    )


# ---------------------------------------------------------------------------
# A second published statement of the same two canals
# ---------------------------------------------------------------------------


def test_the_two_published_geometries_agree():
    """Both canals, described twice by unrelated authors, eleven years apart.

    The reach lengths, bed widths, side slopes, roughness and bed slopes
    here were taken from a 2025 paper for one canal and a 2015 paper for
    the other. Litrico and Fromion published both canals in 2004 in their
    own Tables 1 and 2. Every number agrees.

    That is what licenses taking Corning's target depths from the 2004
    paper - the only published statement of them found for this study -
    without mixing two descriptions of different canals.
    """
    from faircanal.geometry import LITRICO_GEOMETRY_SOURCE

    assert "Table 1" in LITRICO_GEOMETRY_SOURCE

    litrico_wm = (
        (100.0, 1.0, 0.9),
        (1200.0, 1.0, 0.9),
        (400.0, 1.0, 0.8),
        (800.0, 0.8, 0.9),
        (2000.0, 0.8, 0.9),
        (1700.0, 0.8, 0.8),
        (1600.0, 0.6, 0.8),
        (1700.0, 0.6, 0.8),
    )
    litrico_corning = (
        (7000.0, 7.0, 2.1),
        (3000.0, 7.0, 2.1),
        (3000.0, 7.0, 2.1),
        (4000.0, 6.0, 1.9),
        (4000.0, 6.0, 1.9),
        (3000.0, 5.0, 1.7),
        (2000.0, 5.0, 1.7),
        (2000.0, 5.0, 1.7),
    )
    for pools, published, roughness, slope in (
        (WM_POOLS, litrico_wm, 0.014, 0.002),
        (CORNING_POOLS, litrico_corning, 0.02, 1.0e-4),
    ):
        for pool, (length, bed_width, target) in zip(pools, published):
            assert pool.length_m == length, pool.name
            assert pool.bed_width_m == bed_width, pool.name
            assert pool.target_level_m == target, pool.name
            assert pool.side_slope == 1.5, pool.name
            assert pool.manning_n == roughness, pool.name
            assert pool.bed_slope == slope, pool.name
            assert pool.canal_depth_m > pool.target_level_m, pool.name


# ---------------------------------------------------------------------------
# The wave damping
# ---------------------------------------------------------------------------


def test_the_shape_exponent_tends_to_seven_thirds_in_a_wide_channel():
    """``kappa`` is 7/3 minus a term that vanishes as the banks recede.

    A wide shallow channel is where the Manning exponent takes its textbook
    value, so this is the one place the formula can be checked against a
    number nobody had to publish.
    """
    from faircanal.geometry import shape_exponent

    values = []
    for width in (10.0, 100.0, 1000.0, 10000.0):
        pool = TrapezoidalPool(
            name=f"wide-{width:g}",
            length_m=1000.0,
            bed_width_m=width,
            side_slope=1.5,
            canal_depth_m=2.0,
            manning_n=0.02,
            bed_slope=1.0e-4,
        )
        values.append(shape_exponent(pool, 1.0))
    assert values == sorted(values)
    assert values[-1] == pytest.approx(7.0 / 3.0, abs=0.01)
    assert values[0] < 7.0 / 3.0 - 0.1


def test_a_long_enough_pool_reflects_nothing():
    """The reflection factor goes to one as the round trip gets longer.

    Clemmens et al. say it in words - Eq. (3) is the limit of Eq. (24) as
    the pool length grows - and this is that sentence as a number.
    """
    from faircanal.geometry import reflection_factor, resonance_peak_with_reflections

    base = CORNING_POOLS[0]
    factors = []
    for length in (7.0e3, 7.0e4, 7.0e5):
        pool = TrapezoidalPool(
            name=f"long-{length:g}",
            length_m=length,
            bed_width_m=base.bed_width_m,
            side_slope=base.side_slope,
            canal_depth_m=base.canal_depth_m,
            manning_n=base.manning_n,
            bed_slope=base.bed_slope,
            target_level_m=base.target_level_m,
        )
        velocity = 11.0 / flow_area(pool, 2.1)
        factors.append(reflection_factor(pool, 2.1, velocity))
        assert resonance_peak_with_reflections(
            pool, 2.1, velocity
        ) == pytest.approx(
            resonance_peak_height(pool, 2.1, velocity) * factors[-1], rel=1e-12
        )
    assert factors == sorted(factors, reverse=True)
    assert factors[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(factor >= 1.0 for factor in factors)


def test_a_wave_cannot_travel_upstream_against_supercritical_flow():
    from faircanal.geometry import wave_decay_rate

    pool = WM_POOLS[0]
    with pytest.raises(ValueError, match="subcritical"):
        wave_decay_rate(pool, 0.9, 5.0)
    with pytest.raises(ValueError, match="zero flow"):
        wave_decay_rate(pool, 0.9, 0.0)
