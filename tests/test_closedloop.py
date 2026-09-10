"""Permanent tests for the closed loop. Never deleted.

Each piece of the loop has been checked on its own. These check the piece
that only exists once they are wired together, and one of them settles a
question the source leaves open by running it rather than arguing it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from faircanal.benchmarks import (
    HAUGHTON_SCENARIO,
    haughton_design_pool,
    haughton_filter,
    haughton_pool,
)
from faircanal.closedloop import (
    ClosedLoop,
    kalman_gain,
    simulate_closed_loop,
    step_third_order,
)
from faircanal.control import LqWeights, control_law, simulate_first_order
from faircanal.pool import ramp_rate, simulate_level

CANALS = tuple(HAUGHTON_SCENARIO["pool_model_by_canal"])


def source_loop(reading: str = "covariance") -> ClosedLoop:
    return ClosedLoop(
        plant_pools=tuple(haughton_pool(i, 3) for i in CANALS),
        design_pools=tuple(haughton_design_pool(i) for i in CANALS),
        filter_spec=haughton_filter(HAUGHTON_FILTER_ORDER),
        weights=LqWeights(
            q=HAUGHTON_SCENARIO["cost_q"],
            r_reservoir=HAUGHTON_SCENARIO["cost_r_reservoir_gate"],
        ),
        kalman_r1=HAUGHTON_SCENARIO["kalman_r1"],
        kalman_r2=HAUGHTON_SCENARIO["kalman_r2"],
        kalman_reading=reading,
    )


HAUGHTON_FILTER_ORDER = 3


def disturbance_amplitude() -> float:
    """The offtake that changes pool one's level by one unit per minute.

    The source describes its disturbance by the level rate it produces, not
    by its discharge: "a discharge rate that gives a change of one unit per
    minute to the level". So the discharge is that rate divided by the
    level ramp the pool has per unit of water drawn.
    """
    return 1.0 / ramp_rate(haughton_pool(1, 3))[1]


# ---------------------------------------------------------------------------
# The plant, driven one step at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", (1, 2))
def test_stepping_the_plant_matches_simulating_it(index: int):
    """A feedback loop needs the plant one step at a time; the batch
    implementation stays the reference and they must not drift apart."""
    pool = haughton_pool(index, 3)
    steps = 200
    rng = np.random.default_rng(5)
    inflow = rng.normal(size=steps)
    drawn = rng.normal(size=steps)

    batch = simulate_level(pool, inflow, drawn, np.zeros(steps))
    stepped = np.zeros(steps + 1)
    for step in range(steps):
        stepped[step + 1] = step_third_order(pool, stepped, inflow, drawn, step)
    assert stepped == pytest.approx(batch, abs=1e-12)


# ---------------------------------------------------------------------------
# The observer reading the source leaves ambiguous
# ---------------------------------------------------------------------------


def test_the_two_kalman_readings_give_the_documented_numbers():
    assert kalman_gain(1.0, 100.0, "gain") == pytest.approx(10.512492, abs=1e-6)
    assert kalman_gain(1.0, 100.0, "covariance") == pytest.approx(0.095125, abs=1e-6)


def test_reading_the_riccati_solution_as_a_gain_destroys_the_loop():
    """Which reading was meant, settled by running both.

    Section 4.3 calls the same symbol both the solution of a Riccati
    recursion and the correction gain, and the two cannot both be true: the
    recursion gives about 10.5, and any correction gain above two makes an
    observer diverge. Rather than argue it, both are run. One settles; the
    other overflows to infinity within the run.
    """
    steps = 300
    offtake = np.zeros((steps, len(CANALS)))
    initial = np.array(HAUGHTON_SCENARIO["initial_level_used"])

    settling = simulate_closed_loop(source_loop("covariance"), initial, offtake)
    assert np.isfinite(settling.levels).all()
    assert np.abs(settling.levels[-1]).max() < 0.5

    with np.errstate(over="ignore", invalid="ignore"):
        diverging = simulate_closed_loop(source_loop("gain"), initial, offtake)
    # Three hundred steps in, the levels are past 1e200. Whether the run
    # reaches infinity before it ends is a detail of how long it is asked to
    # run; that it grows without bound is the point.
    assert np.abs(diverging.levels).max() > 1e6, (
        "the gain reading no longer diverges, so it is no longer excluded"
    )
    assert np.abs(diverging.levels[-1]).max() > 1e3 * np.abs(
        diverging.levels[len(diverging.levels) // 2]
    ).max(), "the growth is not compounding, so this is not divergence"


# ---------------------------------------------------------------------------
# The design model against the plant it is a stand-in for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", (1, 2))
def test_the_design_model_tracks_the_filtered_plant(index: int):
    """The source's own claim, checked: a first-order model with an extra
    delay stands in for the third-order pool once its input is filtered.

    This is what licenses designing the controller on one model and running
    it on the other, so it is the single most load-bearing approximation in
    the whole assembly. The source shows it as a figure; here it is a
    number, and the number is under ten per cent.
    """
    steps = 200
    numerator, denominator = haughton_filter(HAUGHTON_FILTER_ORDER).coefficients()
    command = np.zeros(steps)
    command[:50] = 1.0
    command[100:150] = -1.0

    plant = simulate_level(
        haughton_pool(index, 3),
        np.zeros(steps),
        signal.lfilter(numerator, denominator, command),
        np.zeros(steps),
    )
    design = simulate_first_order(
        [haughton_design_pool(index)],
        np.zeros((steps, 1)),
        -command.reshape(steps, 1),
        np.zeros(1),
    )[:, 0]

    scale = np.abs(plant).max()
    assert np.abs(plant - design).max() < 0.10 * scale
    assert abs(plant[-1]) < 0.01 * scale, "the plant must return to its set-point"


# ---------------------------------------------------------------------------
# The loop as a whole
# ---------------------------------------------------------------------------


def test_the_loop_settles_after_a_set_point_change():
    """A step in the set-point is corrected and stays corrected."""
    steps = 300
    result = simulate_closed_loop(
        source_loop(),
        np.array(HAUGHTON_SCENARIO["initial_level_used"]),
        np.zeros((steps, len(CANALS))),
    )
    assert np.abs(result.levels[150:]).max() < 0.2


def test_the_filter_really_sits_between_controller_and_plant():
    """What the controller asks for and what the canal receives differ.

    The gap between them is the whole subject of this study, so a loop in
    which they coincided would be the wrong loop, however well it behaved.
    """
    steps = 200
    offtake = np.zeros((steps, len(CANALS)))
    offtake[50:150, 0] = disturbance_amplitude()
    result = simulate_closed_loop(source_loop(), np.zeros(len(CANALS)), offtake)

    assert np.abs(result.commanded - result.applied).max() > 1.0
    assert np.abs(result.offtake_applied - offtake).max() > 1.0
    # And the filter conserves what it delays.
    assert result.offtake_applied[:, 0].sum() == pytest.approx(
        offtake[:, 0].sum(), rel=0.06
    )


def test_the_flows_balance_while_an_offtake_runs():
    """In steady state the controller replaces exactly what is drawn.

    The inflow and the offtake reach the level through different
    coefficients, so balance is not "inflow equals offtake" but "inflow
    times its ramp equals offtake times its". Getting that wrong would show
    up as a level drifting without limit rather than holding an offset.
    """
    steps = 600
    amplitude = disturbance_amplitude()
    offtake = np.zeros((steps, len(CANALS)))
    offtake[250:450, 0] = amplitude
    result = simulate_closed_loop(source_loop(), np.zeros(len(CANALS)), offtake)

    inflow_ramp, outflow_ramp = ramp_rate(haughton_pool(1, 3))
    supplied = result.applied[300:440, 0].mean() * inflow_ramp
    drawn = result.offtake_applied[300:440, 0].mean() * outflow_ramp
    assert supplied == pytest.approx(drawn, rel=0.02)

    # Holding an offset is allowed; drifting is not. The source's controller
    # has no integral action, so an offset established during the transition
    # is never recovered - which is a property to record, not a fault.
    early = result.levels[300:340, 0].mean()
    late = result.levels[400:440, 0].mean()
    assert abs(late - early) < 0.5, "the level is drifting, not holding"


def test_an_announced_offtake_is_answered_before_it_arrives():
    """Feed-forward, end to end: the loop acts on what it has been told."""
    steps = 400
    offtake = np.zeros((steps, len(CANALS)))
    offtake[200:300, 0] = disturbance_amplitude()
    result = simulate_closed_loop(source_loop(), np.zeros(len(CANALS)), offtake)
    assert np.abs(result.commanded[150:200]).max() > 1.0


def test_a_loop_with_mismatched_model_orders_is_refused():
    with pytest.raises(ValueError, match="third-order"):
        ClosedLoop(
            plant_pools=(haughton_design_pool(1),),
            design_pools=(haughton_design_pool(1),),
            filter_spec=haughton_filter(3),
            weights=LqWeights(),
        )
    with pytest.raises(ValueError, match="first-order"):
        ClosedLoop(
            plant_pools=(haughton_pool(1, 3),),
            design_pools=(haughton_pool(1, 3),),
            filter_spec=haughton_filter(3),
            weights=LqWeights(),
        )
