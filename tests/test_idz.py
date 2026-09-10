"""Permanent tests for the integrator-delay-zero layer. Never deleted.

The experiments run at flows nobody has tabulated, so the pool parameters
have to be computed rather than copied. That makes this module the one
place where a wrong transcription of an appendix would silently move every
result, and it is why the published tables are kept here as data: they are
thirty-two independent chances for the construction to be caught out.
"""

from __future__ import annotations

import math

import pytest

from faircanal.geometry import (
    CORNING_POOLS,
    WM_POOLS,
    TrapezoidalPool,
    flow_area,
    normal_depth,
    top_width,
    wave_celerity,
    wave_decay_rate,
)
from faircanal.idz import (
    LITRICO_MAX_RELATIVE_ERRORS,
    LITRICO_TABLES,
    IdzParams,
    _part_areas,
    _part_gains,
    _surface_decay,
    backwater_split,
    decay_implied_by_published_range,
    depth_gradient,
    froude_number,
    friction_slope,
    idz_from_geometry,
    reflection_amplification,
    round_trip_decay,
)

CANALS = {"wm": WM_POOLS, "corning": CORNING_POOLS}

#: How far the geometry route may sit from the printed value, in per cent.
#: These are measurements, not aspirations: they are what the construction
#: currently achieves, rounded up to the next sensible figure, so that any
#: drift is caught rather than absorbed.
TOLERANCES = {
    "wm": {"delay": 0.2, "area": 6.0, "p21": 2.0, "p22": 0.1},
    "corning": {"delay": 0.02, "area": 0.1, "p21": 0.7, "p22": 0.2},
}


def rows_and_pools(canal: str, flow: str):
    return list(zip(LITRICO_TABLES[(canal, flow)], CANALS[canal]))


# ---------------------------------------------------------------------------
# The headline: two routes to the same numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canal", sorted(CANALS))
@pytest.mark.parametrize("flow", ("low", "high"))
def test_the_geometry_route_reproduces_the_published_tables(canal: str, flow: str):
    """Thirty-two pools, four parameters each, from geometry alone.

    Nothing about this study rests on a coefficient that was copied. The
    delay, the backwater area and the two feedthrough gains are all
    computed from the section, the slopes, the roughness, the discharge and
    the target depth, and here they are put next to what the source
    printed for the same pools.

    The delay is reproduced to about a tenth of a per cent everywhere. The
    area is exact to a twentieth of a per cent on the flat canal and up to
    six per cent out on the steep one - which is the canal the source
    itself finds hardest, reporting in its own Table 8 that its model sits
    four per cent from theory there.
    """
    limits = TOLERANCES[canal]
    for row, pool in rows_and_pools(canal, flow):
        got = idz_from_geometry(pool, row.discharge_m3_s)
        for name, computed, printed in (
            ("delay", got.delay_s, row.delay_model_s),
            ("area", got.area_m2, row.area_model_m2),
            ("p21", got.p21, row.p21),
            ("p22", got.p22, row.p22),
        ):
            error = 100.0 * abs(computed - printed) / printed
            assert error <= limits[name], (
                f"{canal} {flow} pool {row.pool}: {name} came out {computed:.6g} "
                f"against a published {printed:.6g}, which is {error:.3f} per cent"
            )


def test_the_bed_slope_reading_of_the_backwater_profile_is_refuted():
    """Which slope Eq. (41) means, settled by running both.

    The equation prints the backed-up surface as a straight line of slope
    ``S_b`` and the sentence under it calls the slope ``S_X``; Eq. (42) only
    makes sense with ``S_X``. Rather than argue it, both are computed on
    the flat canal, where the published areas are otherwise reproduced to a
    twentieth of a per cent. One reading keeps that agreement and the other
    loses it by more than an order of magnitude.
    """
    worst = {"surface": 0.0, "bed": 0.0}
    for row, pool in rows_and_pools("corning", "high"):
        for reading in worst:
            got = idz_from_geometry(
                pool, row.discharge_m3_s, profile_slope=reading
            )
            error = 100.0 * abs(got.area_m2 - row.area_model_m2) / row.area_model_m2
            worst[reading] = max(worst[reading], error)
    assert worst["surface"] < 0.1
    assert worst["bed"] > 5.0, (
        "the bed-slope reading no longer misses the published areas, so this "
        "question is no longer settled by measurement"
    )


def test_the_two_papers_agree_on_the_decay_rate():
    """Litrico's Eq. (65) and Clemmens's Eqs (25)-(29) are one formula.

    They are written in different notation from different starting points
    in papers eleven years apart, and for uniform flow they agree to the
    last bit. Reproducing that is a check on both transcriptions at once,
    and it is the reason the wave damping can be trusted at all.
    """
    for pools, discharge in ((WM_POOLS, 1.0), (CORNING_POOLS, 6.0)):
        for pool in pools:
            depth = normal_depth(pool, discharge)
            velocity = discharge / flow_area(pool, depth)
            litrico = _surface_decay(pool, depth, discharge, 0.0)
            clemmens = wave_decay_rate(pool, depth, velocity)
            assert litrico == pytest.approx(clemmens, rel=1e-12)


def test_the_geometry_damping_agrees_with_the_published_ranges():
    """A third route to the wave damping, using no appendix at all.

    Tables 4 to 7 print the smallest and largest value the high-frequency
    gain takes. Those are the trough and the crest of one comb, so each of
    them alone gives back the round-trip decay. The two readings of the
    same row agree with each other to within five hundredths, which is what
    says the comb is the right picture.

    The geometry route then sits above them at every one of the sixteen
    rows, keeping between one and twenty-five hundredths more of the wave
    per round trip. The worst case is the first pool - seven kilometres
    long and the deepest under backwater - where the geometry route keeps
    0.41 of the wave and the published range says 0.16.

    That gap is a property of the model, not a tolerance to be tuned. It is
    recorded here so that the experiments can be run both ways.
    """
    gaps = []
    for flow in ("low", "high"):
        for row, pool in rows_and_pools("corning", flow):
            crest, trough = decay_implied_by_published_range(pool, row)
            assert crest == pytest.approx(trough, abs=0.05), (
                f"corning {flow} pool {row.pool}: the crest and the trough of the "
                f"published range give {crest:.3f} and {trough:.3f} for the same "
                f"decay, so they are not two ends of one comb"
            )
            gaps.append(round_trip_decay(pool, row.discharge_m3_s) - crest)
    assert all(gap > 0.0 for gap in gaps), "the sign of the gap has changed"
    assert max(gaps) < 0.30, f"the gap has grown: worst is {max(gaps):.3f}"


# ---------------------------------------------------------------------------
# The pieces the construction is made of
# ---------------------------------------------------------------------------


def test_a_pool_at_normal_depth_is_all_uniform():
    """No backwater means no backed-up part, by Eq. (40)."""
    pool = CORNING_POOLS[0]
    discharge = 6.0
    depth = normal_depth(pool, discharge)
    split = backwater_split(pool, depth, discharge)
    assert split.uniform_length_m == pytest.approx(pool.length_m, rel=1e-9)
    assert split.backwater_length_m == pytest.approx(0.0, abs=1e-6)
    assert depth_gradient(pool, depth, discharge) == pytest.approx(0.0, abs=1e-12)


def test_a_deeper_pool_is_more_backed_up():
    """Holding a higher level pushes the backed-up part further upstream.

    The steep canal is used because its pools are only partly backed up at
    their target level, so there is something to move; the flat canal's are
    backed up over their whole length at every published operating point
    and the answer would be the pool length whatever the depth.
    """
    pool = WM_POOLS[4]
    discharge = 0.4
    lengths = [
        backwater_split(pool, depth, discharge).backwater_length_m
        for depth in (0.6, 0.7, 0.8, 0.9)
    ]
    assert lengths == sorted(lengths)
    assert lengths[-1] > 1.9 * lengths[0]
    assert 0.0 < lengths[-1] < pool.length_m


def test_a_short_part_stores_its_surface_area():
    """Equations (49) and (50) both tend to ``T_0 x`` for a short reach.

    The source states this in words just after Eq. (12) - that the area is
    close to the surface area of the pool when the exponent is small - and
    it is the one independent statement of what those two equations mean.
    """
    pool = CORNING_POOLS[0]
    discharge = 8.0
    depth = normal_depth(pool, discharge)
    errors = []
    for length in (100.0, 10.0, 1.0):
        down, up = _part_areas(pool, depth, discharge, 0.0, length)
        surface = top_width(pool, depth) * length
        assert down == pytest.approx(surface, rel=0.02)
        assert up == pytest.approx(surface, rel=0.02)
        errors.append(abs(down - surface) / surface)
    # And the approximation is first order in the length, so a tenth of the
    # reach leaves a tenth of the error. That is the shape of the statement,
    # not just its value at one length.
    assert errors[1] == pytest.approx(errors[0] / 10.0, rel=0.05)
    assert errors[2] == pytest.approx(errors[1] / 10.0, rel=0.05)


def test_a_long_reach_reaches_the_published_asymptote():
    """``p22`` of a very long pool is Eq. (3) of the other paper.

    Clemmens et al. say where their Eq. (3) came from: "modifying their
    equation for p22 for a pool of infinite length". Taking Eq. (64) to
    that limit has to land on ``1 / (B (c - v))``, and it does. This is
    what fixes the reading of Eq. (64), whose printed form is ambiguous
    about what the square root covers.
    """
    pool = CORNING_POOLS[0]
    discharge = 8.0
    depth = normal_depth(pool, discharge)
    velocity = discharge / flow_area(pool, depth)
    asymptote = 1.0 / (top_width(pool, depth) * (wave_celerity(pool, depth) - velocity))
    *_, p22 = _part_gains(pool, depth, discharge, 0.0, 400_000.0)
    assert p22 == pytest.approx(asymptote, rel=1e-6)


def test_a_zero_length_reach_at_rest_is_the_still_water_gain():
    """And the other limit of Eq. (64): ``1 / (T_0 C_0)``.

    Zero length and a vanishing Froude number, where the two directions of
    travel are the same speed and the expression has to collapse to the
    still-water value. Both limits are needed: at zero length alone the
    gain still carries the asymmetry between upstream and downstream.
    """
    pool = CORNING_POOLS[0]
    depth = 2.1
    still = 1.0 / (top_width(pool, depth) * wave_celerity(pool, depth))
    *_, p22 = _part_gains(pool, depth, 1.0e-4, 0.0, 0.0)
    assert p22 == pytest.approx(still, rel=1e-4)

    # At a real discharge it sits between the two one-way gains instead.
    velocity = 8.0 / flow_area(pool, depth)
    celerity = wave_celerity(pool, depth)
    width = top_width(pool, depth)
    *_, p22 = _part_gains(pool, depth, 8.0, 0.0, 0.0)
    assert 1.0 / (width * (celerity + velocity)) < p22
    assert p22 < 1.0 / (width * (celerity - velocity))


def test_the_friction_slope_at_normal_depth_is_the_bed_slope():
    """What normal depth means, checked rather than assumed."""
    for pool in (CORNING_POOLS[2], WM_POOLS[4]):
        discharge = 5.0 if pool in CORNING_POOLS else 0.5
        depth = normal_depth(pool, discharge)
        assert friction_slope(pool, depth, discharge) == pytest.approx(
            pool.bed_slope, rel=1e-9
        )


def test_the_flow_stays_subcritical_at_every_published_operating_point():
    """A Froude number above one would put every formula here out of scope."""
    for (canal, flow), rows in LITRICO_TABLES.items():
        for row, pool in zip(rows, CANALS[canal]):
            froude = froude_number(pool, pool.target_level_m, row.discharge_m3_s)
            assert 0.0 < froude < 0.6, f"{canal} {flow} pool {row.pool}: F = {froude:.3f}"


# ---------------------------------------------------------------------------
# The transcription itself
# ---------------------------------------------------------------------------


def test_the_published_flows_step_down_along_each_canal():
    """Every table is one cascade, so the discharge only falls downstream.

    A transposed pair of rows would show up here and nowhere else.
    """
    for key, rows in LITRICO_TABLES.items():
        flows = [row.discharge_m3_s for row in rows]
        assert flows == sorted(flows, reverse=True), key
        assert len(rows) == 8


def test_every_published_range_brackets_something():
    """Each printed pair is an interval, and each gain is positive."""
    for key, rows in LITRICO_TABLES.items():
        for row in rows:
            for low, high in (row.p21_range, row.p22_range):
                assert 0.0 < low < high, f"{key} pool {row.pool}"
            assert row.p21 > 0.0 and row.p22 > 0.0
            assert row.delay_model_s > 0.0 and row.area_model_m2 > 0.0


def test_table_eight_is_not_a_summary_of_the_tables_before_it():
    """A discrepancy inside the source, recorded so nobody "fixes" it.

    Table 8 prints the largest relative error of the delay and of the
    backwater area for each canal and flow. Recomputing those from the
    columns of Tables 4 to 7 reproduces the delay row well - exactly for
    the flat canal at high flow, and within about a tenth of a point
    elsewhere - but not the area row, which comes out at 1.0 and 0.9 per
    cent for the steep canal where Table 8 says 3.26 and 4.03.

    So Table 8's area column is measured against some other quantity than
    the ``A_d`` column printed beside it - most likely the backwater area
    of Eq. (12), which needs the full backwater curve rather than the
    two-line approximation. The transcription here follows Tables 4 to 7,
    which is what this study uses; this test exists so that a later reader
    who notices the mismatch finds it already accounted for rather than
    correcting a transcription that is not wrong.
    """
    computed = {}
    for key, rows in LITRICO_TABLES.items():
        computed[key] = {
            "delay": max(
                100.0 * abs(row.delay_model_s - row.delay_theory_s) / row.delay_theory_s
                for row in rows
            ),
            "area": max(
                100.0 * abs(row.area_model_m2 - row.area_theory_m2) / row.area_theory_m2
                for row in rows
            ),
        }

    for key, values in computed.items():
        printed = LITRICO_MAX_RELATIVE_ERRORS[key]["delay"]
        assert values["delay"] == pytest.approx(printed, abs=0.09), (
            f"{key}: the delay errors in Tables 4-7 no longer agree with Table 8"
        )

    assert computed[("corning", "high")]["delay"] == pytest.approx(0.31, abs=0.005)
    assert computed[("wm", "low")]["area"] < 1.5
    assert LITRICO_MAX_RELATIVE_ERRORS[("wm", "low")]["area"] > 3.0
    assert computed[("wm", "high")]["area"] < 1.5
    assert LITRICO_MAX_RELATIVE_ERRORS[("wm", "high")]["area"] > 3.0


def test_the_low_flow_tables_are_the_published_test_scenario():
    """The steep canal's low-flow row is ASCE test 1-1, reached separately.

    ``WM_HEADING_FLOW`` and ``WM_CHECK_FLOWS`` were transcribed from a
    different paper entirely, and the discharges of Table 4 are the same
    cascade: the heading flow followed by the flow past each check. Two
    unrelated transcriptions landing on the same eight numbers is worth
    more than either of them alone.
    """
    from faircanal.geometry import WM_CHECK_FLOWS, WM_HEADING_FLOW

    published = [row.discharge_m3_s for row in LITRICO_TABLES[("wm", "low")]]
    assembled = [WM_HEADING_FLOW, *WM_CHECK_FLOWS[:-1]]
    assert published == pytest.approx(assembled, abs=1e-9)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_pool_with_no_target_level_and_no_depth_is_refused():
    pool = TrapezoidalPool(
        name="nameless",
        length_m=1000.0,
        bed_width_m=5.0,
        side_slope=1.5,
        canal_depth_m=2.0,
        manning_n=0.02,
        bed_slope=1.0e-4,
    )
    with pytest.raises(ValueError, match="target level"):
        idz_from_geometry(pool, 5.0)


def test_a_supercritical_flow_is_refused():
    pool = CORNING_POOLS[0]
    with pytest.raises(ValueError, match="subcritical"):
        depth_gradient(pool, 0.05, 40.0)


def test_a_negative_discharge_is_refused():
    with pytest.raises(ValueError, match="positive"):
        idz_from_geometry(CORNING_POOLS[0], -1.0)


def test_an_unknown_profile_reading_is_refused():
    with pytest.raises(ValueError, match="profile_slope"):
        backwater_split(CORNING_POOLS[0], 2.1, 11.0, profile_slope="whatever")


def test_the_parameters_refuse_to_hold_nonsense():
    with pytest.raises(ValueError, match="backwater area"):
        IdzParams(
            name="bad",
            discharge_m3_s=1.0,
            depth_m=2.0,
            delay_s=10.0,
            area_m2=0.0,
            p21=0.01,
            p22=0.01,
        )
    with pytest.raises(ValueError, match="delay"):
        IdzParams(
            name="bad",
            discharge_m3_s=1.0,
            depth_m=2.0,
            delay_s=-1.0,
            area_m2=100.0,
            p21=0.01,
            p22=0.01,
        )


# ---------------------------------------------------------------------------
# What the numbers say about these canals
# ---------------------------------------------------------------------------


def test_low_flow_rings_harder_than_high_flow():
    """The source's own statement, reproduced from geometry.

    Clemmens et al. write that "the resonance peak at low discharges can be
    significantly higher than that predicted by Eq. (3)", because a pool
    held at its target level while carrying little water is deeply backed
    up and a backed-up reach hardly damps a wave. The uniform-flow reading
    of the damping says the opposite, so this test is also what rules that
    reading out.
    """
    for index, pool in enumerate(CORNING_POOLS):
        low = LITRICO_TABLES[("corning", "low")][index].discharge_m3_s
        high = LITRICO_TABLES[("corning", "high")][index].discharge_m3_s
        assert reflection_amplification(pool, low) > reflection_amplification(
            pool, high
        ), pool.name


def test_only_the_first_pool_of_the_steep_canal_rings():
    """"Pools in this canal do not experience oscillations, except in the
    first pool" - reproduced, not assumed.

    The steep canal's first reach is a hundred metres long and comes out
    with a resonance many times its asymptote; every other reach comes out
    with none at all. That is a published qualitative statement about a
    canal, recovered from nothing but its geometry.
    """
    for flow in ("low", "high"):
        rows = LITRICO_TABLES[("wm", flow)]
        first = reflection_amplification(WM_POOLS[0], rows[0].discharge_m3_s)
        assert first > 5.0, f"wm {flow} pool 1 came out at {first:.3f}"
        for row, pool in list(zip(rows, WM_POOLS))[1:]:
            rest = reflection_amplification(pool, row.discharge_m3_s)
            assert rest < 1.01, f"wm {flow} pool {row.pool} came out at {rest:.3f}"
