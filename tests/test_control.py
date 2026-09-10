"""Permanent tests for the controller design. Never deleted.

Two things are being guarded. The first is the transcription of the
source's parameter sweep, which is checked structurally rather than
against numbers nobody published: the recursion has properties a correct
transcription must have and a mistyped one will not.

The second is the arbiter. The whole plan for reproducing this controller
rests on solving the source's control problem directly and comparing, so
the solver of that problem has to be right before it can judge anything.
It is checked the only way an optimiser can be: no perturbation of its
answer is allowed to be cheaper.
"""

from __future__ import annotations

import numpy as np
import pytest

from faircanal.benchmarks import HAUGHTON_SCENARIO, haughton_design_pool, haughton_pool
from faircanal.control import (
    LqWeights,
    design_parameters,
    offtake_to_source_disturbance,
    simulate_first_order,
    solve_lq_trajectory,
)
from faircanal.pool import simulate_level

#: The five pool network of the source, most downstream pool first.
NETWORK = tuple(HAUGHTON_SCENARIO["pool_model_by_canal"])


def source_network() -> list:
    return [haughton_design_pool(i) for i in NETWORK]


# ---------------------------------------------------------------------------
# The parameter sweep
# ---------------------------------------------------------------------------


def test_a_single_pool_needs_no_sweep():
    """With one pool the recursion does not run and the closed forms apply."""
    pool = haughton_design_pool(1)
    weights = LqWeights()
    design = design_parameters([pool], weights)

    assert design.gamma == pytest.approx([weights.q])
    assert design.b_hat == pytest.approx([pool.b[0]])
    assert design.r_scaled == pytest.approx(weights.r_reservoir / pool.b[0] ** 2)

    gamma = weights.q
    expected_x = -gamma / 2 + np.sqrt(gamma * design.r_scaled + gamma**2 / 4)
    assert design.x == pytest.approx(expected_x)
    assert design.g == pytest.approx(expected_x / (expected_x + gamma))


def test_gamma_falls_strictly_going_upstream():
    """Each step combines two positive numbers into something below both.

    gamma_i = gamma_{i-1} q_i / (gamma_{i-1} + q_i) is half the harmonic
    mean, so it is strictly less than either input. A transcription that
    turned the recursion into a sum or a product would break this
    immediately, and would otherwise produce a plausible-looking controller.
    """
    design = design_parameters(source_network(), LqWeights())
    for i in range(1, len(design.gamma)):
        assert design.gamma[i] < design.gamma[i - 1]
        assert design.gamma[i] < design.q_scaled[i]
        assert design.gamma[i] > 0.0


def test_the_preview_weight_is_a_discount():
    """g must lie strictly between zero and one.

    It weights disturbances that have not happened yet. At zero the
    controller ignores what it has been told is coming; at one it treats a
    disturbance an hour away as if it were happening now.
    """
    design = design_parameters(source_network(), LqWeights())
    assert 0.0 < design.g < 1.0


def test_a_heavier_reservoir_penalty_makes_the_controller_more_patient():
    """Raising r must raise both X and the preview weight.

    A costlier reservoir flow means acting earlier and more gently on
    previewed disturbances, so g rises. If the sign of that dependence came
    out backwards, the controller would still run and would be tuned the
    wrong way.
    """
    pools = source_network()
    cheap = design_parameters(pools, LqWeights(r_reservoir=0.3))
    dear = design_parameters(pools, LqWeights(r_reservoir=3.0))
    assert dear.x > cheap.x
    assert dear.g > cheap.g


def test_weights_outside_the_structure_are_refused():
    """The source's theorem covers one shape of cost and no other."""
    with pytest.raises(ValueError, match="rho = 0"):
        LqWeights(rho=0.1)
    with pytest.raises(ValueError, match="must be positive"):
        LqWeights(r_reservoir=0.0)


def test_third_order_pools_are_refused_by_the_design_model():
    """The controller is designed on the first-order approximation."""
    with pytest.raises(ValueError, match="first order"):
        simulate_first_order(
            [haughton_pool(1, 3)], np.zeros((5, 1)), np.zeros((5, 1)), np.zeros(1)
        )


# ---------------------------------------------------------------------------
# Sign conventions across the package boundary
# ---------------------------------------------------------------------------


def test_the_offtake_conversion_is_exactly_a_sign():
    offtake = np.array([0.0, 1.0, 2.5, -0.5])
    converted = offtake_to_source_disturbance(offtake)
    assert converted == pytest.approx(-offtake)
    assert offtake_to_source_disturbance(converted) == pytest.approx(offtake)


def test_a_withdrawal_lowers_the_level_in_both_conventions():
    """The design model and the plant model must agree about which way is down.

    They are written in opposite conventions on purpose - the design model
    reproduces the source, the plant model reproduces the physics - so the
    one place they meet is worth a test. If the conversion were dropped, a
    withdrawal would raise the level in the controller's model and lower it
    in the plant, and the controller would push the wrong way.
    """
    steps = 400
    pool = haughton_design_pool(1)
    offtake = np.zeros(steps)
    offtake[100:] = 1.0

    design_level = simulate_first_order(
        [pool],
        np.zeros((steps, 1)),
        offtake_to_source_disturbance(offtake).reshape(steps, 1),
        np.zeros(1),
    )
    plant_level = simulate_level(
        haughton_pool(1, 3), np.zeros(steps), np.zeros(steps), offtake
    )

    assert design_level[-1, 0] < 0.0
    assert plant_level[-1] < 0.0


def test_a_gate_moves_water_from_one_pool_to_the_next():
    """Driving one gate must fill the pool below it and drain the pool above.

    u_1 is pool 1's inflow and pool 2's outflow at the same time - it is
    one gate. If both signs are right, opening it raises pool 1, lowers
    pool 2 and leaves everything further upstream alone.
    """
    pools = source_network()
    steps, n = 60, len(pools)
    u = np.zeros((steps, n))
    u[:, 0] = 1.0
    y = simulate_first_order(pools, u, np.zeros((steps, n)), np.zeros(n))

    assert y[-1, 0] > 0.0, "the pool below the gate must fill"
    assert y[-1, 1] < 0.0, "the pool above the gate must drain"
    assert y[-1, 2:] == pytest.approx(np.zeros(n - 2), abs=1e-12), (
        "pools further upstream are not connected to this gate"
    )


def test_the_downstream_gate_is_held_fixed():
    """Pool 1 has no outflow term, because the source fixes that flow.

    Letting the last gate move would hand the controller a free drain and
    quietly improve every result in the study. The property is that
    nothing except pool 1's own inflow and its own offtake can change pool
    1's level.
    """
    pools = source_network()
    steps, n = 80, len(pools)
    rng = np.random.default_rng(4282)

    u = rng.normal(size=(steps, n))
    u[:, 0] = 0.0  # everything except pool 1's inflow
    d = rng.normal(size=(steps, n))
    d[:, 0] = 0.0  # everything except pool 1's offtake

    y = simulate_first_order(pools, u, d, np.zeros(n))
    assert y[:, 0] == pytest.approx(np.zeros(steps + 1), abs=1e-12), (
        "something other than pool 1's own inflow and offtake moved its level"
    )


# ---------------------------------------------------------------------------
# The arbiter
# ---------------------------------------------------------------------------


def test_the_design_model_is_linear():
    pools = source_network()
    steps, n = 80, len(pools)
    rng = np.random.default_rng(2203)
    u1, u2 = rng.normal(size=(steps, n)), rng.normal(size=(steps, n))
    d1, d2 = rng.normal(size=(steps, n)), rng.normal(size=(steps, n))
    y0a, y0b = rng.normal(size=n), rng.normal(size=n)

    combined = simulate_first_order(pools, u1 + 2 * u2, d1 + 2 * d2, y0a + 2 * y0b)
    separate = simulate_first_order(pools, u1, d1, y0a) + 2 * simulate_first_order(
        pools, u2, d2, y0b
    )
    assert combined == pytest.approx(separate, abs=1e-9)


def test_no_perturbation_of_the_lq_solution_is_cheaper():
    """The arbiter has to be optimal before it can judge anything.

    The problem is an unconstrained convex quadratic, so the minimiser is
    unique and every direction out of it costs more. Random directions at
    several magnitudes is a weak test individually and a strong one in
    aggregate: a solver that stopped early, or minimised the wrong cost,
    fails it.
    """
    pools = [haughton_design_pool(i) for i in (1, 2)]
    weights = LqWeights()
    steps = 250
    d = np.zeros((steps, 2))
    u_star, _, cost_star = solve_lq_trajectory(
        pools, weights, np.array([-1.0, 1.0]), d
    )

    rng = np.random.default_rng(16575)
    for scale in (1e-3, 1e-2, 1e-1, 1.0):
        for _ in range(5):
            perturbed = u_star + scale * rng.normal(size=u_star.shape)
            y = simulate_first_order(pools, perturbed, d, np.array([-1.0, 1.0]))
            cost = weights.q * np.sum(y[1:] ** 2) + weights.r_reservoir * np.sum(
                perturbed[:, 1] ** 2
            )
            assert cost >= cost_star - 1e-9, (
                f"a perturbation of size {scale} was cheaper than the optimum"
            )


def test_the_lq_solution_brings_the_network_to_rest():
    """An infinite-horizon cost on the levels means the levels must settle."""
    pools = [haughton_design_pool(i) for i in (1, 2)]
    steps = 250
    _, y, _ = solve_lq_trajectory(
        pools, LqWeights(), np.array([-1.0, 1.0]), np.zeros((steps, 2))
    )
    assert np.abs(y[-1]).max() < 1e-3


def test_the_horizon_is_long_enough_to_stand_in_for_an_infinite_one():
    """The source's problem is infinite-horizon; this one is not.

    What makes the substitution legitimate is that the first inputs stop
    moving once the horizon is long enough. That is checked here rather
    than asserted, because too short a horizon produces a controller that
    is optimal for a deadline nobody imposed.
    """
    pools = [haughton_design_pool(i) for i in (1, 2)]
    weights = LqWeights()
    y0 = np.array([-1.0, 1.0])

    shorter, *_ = solve_lq_trajectory(pools, weights, y0, np.zeros((400, 2)))
    longer, *_ = solve_lq_trajectory(pools, weights, y0, np.zeros((600, 2)))
    assert shorter[:50] == pytest.approx(longer[:50], abs=1e-8)


def test_the_horizon_requirement_was_measured_not_guessed():
    """How long is long enough, in numbers.

    Doubling the horizon changes the first fifty inputs by 1.8e-05 going
    from 200 steps to 300, by 9e-09 going to 400, and by 5e-12 going to
    600, where it reaches the floor of the arithmetic. So 400 steps is
    enough for anything this study reports and 600 - the length the source
    simulated - is comfortably beyond it. A horizon of 200 is not, which is
    why this is a test and not a remark.
    """
    pools = [haughton_design_pool(i) for i in (1, 2)]
    weights = LqWeights()
    y0 = np.array([-1.0, 1.0])

    heads = {}
    for horizon in (200, 300, 400, 600):
        u, *_ = solve_lq_trajectory(pools, weights, y0, np.zeros((horizon, 2)))
        heads[horizon] = u[:50]

    assert np.abs(heads[300] - heads[200]).max() > 1e-6, (
        "a 200 step horizon must visibly differ, or this test proves nothing"
    )
    assert np.abs(heads[400] - heads[300]).max() < 1e-7
    assert np.abs(heads[600] - heads[400]).max() < 1e-9


def test_a_previewed_withdrawal_is_answered_before_it_starts():
    """Feed-forward: the controller acts on what it has been told is coming.

    The source's whole premise is that offtakes are announced in advance.
    If the optimal input were flat until the withdrawal began, there would
    be no feed-forward and no reason for the announcement.
    """
    pools = [haughton_design_pool(i) for i in (1, 2)]
    steps = 300
    offtake = np.zeros((steps, 2))
    offtake[150:200, 0] = 1.0
    d = offtake_to_source_disturbance(offtake)

    u, _, _ = solve_lq_trajectory(pools, LqWeights(), np.zeros(2), d)
    before = np.abs(u[100:150]).max()
    assert before > 1e-6, "the controller ignored a disturbance it was shown"
