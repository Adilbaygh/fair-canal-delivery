"""The closed loop: plant, filter, observer and controller, running together.

This is where the pieces meet. The controller was designed on a first-order
model and is applied to the third-order one; the filter sits between the
controller and the plant and also in front of every planned offtake; and a
scalar observer per gate reconciles the two models. Reproducing the source
study's published response is what says the assembly is right, and it is
the only thing that can say so, because each piece on its own has already
been checked and each piece on its own is not the point.

What runs in what order
-----------------------
At step ``t``:

1. the controller reads the **a priori** level estimate, the commanded
   inputs it has issued before, and the offtakes it has been told are
   coming, and issues ``u[t]``;
2. that command is low-pass filtered, and so is the planned offtake;
3. the filtered signals drive the third-order plant, giving ``y[t+1]``;
4. the observer corrects its estimate with the measurement and predicts
   the next one.

The controller uses the estimate from the previous step rather than the
one just corrected, which the source states outright and motivates: it
leaves a sample time for information to travel along the string of gates.

Two open readings live here
---------------------------
The source's observer is described in a way that admits two readings, and
its baseline is described with two contradictory initial conditions.
Neither is resolved by argument; both are carried as explicit options, and
the one that reproduces the published response is the one that was meant.
That is what :func:`simulate_closed_loop` is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from faircanal.control import LinearControlLaw, LqWeights, control_law
from faircanal.delivery import FilterSpec
from faircanal.pool import PoolParams

__all__ = [
    "KALMAN_READINGS",
    "kalman_gain",
    "ClosedLoop",
    "ClosedLoopResult",
    "simulate_closed_loop",
    "step_third_order",
]


#: The two ways the source's observer can be read. See :func:`kalman_gain`.
KALMAN_READINGS = ("covariance", "gain")


def kalman_gain(r1: float, r2: float, reading: str = "covariance") -> float:
    """The observer gain, under one of the two readings of the source.

    Section 4.3 writes the correction as ``y_hat = y_hat + L (y - y_hat)``
    and says ``L`` solves ``L = L - L^2/(L + R_2) + R_1``. Those two
    statements cannot both be about the same quantity. Solving the
    recursion gives ``L = (R_1 + sqrt(R_1^2 + 4 R_1 R_2)) / 2``, which for
    the published weights is about 10.5 - impossible as a correction gain,
    since anything above two makes the observer diverge.

    The reading that makes sense is that the recursion is the usual scalar
    Riccati equation for the estimate's variance, and the correction gain
    is ``P / (P + R_2)`` - about 0.095 here. Both readings are available so
    that the choice is settled by reproducing the published response rather
    than by argument.
    """
    if reading not in KALMAN_READINGS:
        raise ValueError(f"reading must be one of {KALMAN_READINGS}, got {reading!r}")
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("both variances must be positive")

    variance = (r1 + np.sqrt(r1**2 + 4.0 * r1 * r2)) / 2.0
    if reading == "gain":
        return float(variance)
    return float(variance / (variance + r2))


def step_third_order(
    pool: PoolParams,
    levels: np.ndarray,
    inflow: np.ndarray,
    drawn: np.ndarray,
    step: int,
) -> float:
    """One step of the third-order pool model, from stored histories.

    The same recursion as :func:`faircanal.pool.simulate_level`, written to
    advance a single step so the plant can be driven inside a feedback
    loop. ``drawn`` is everything leaving at the downstream end - the
    outflow through the gate plus the offtake - and a positive value lowers
    the level.

    ``test_stepping_the_plant_matches_simulating_it`` checks this against
    the batch implementation, which stays the reference.
    """

    def past(signal_array: np.ndarray, index: int) -> float:
        return 0.0 if index < 0 else float(signal_array[index])

    def level(index: int) -> float:
        return float(levels[0]) if index < 0 else float(levels[index])

    if pool.order == 1:
        return (
            level(step)
            + pool.b[0] * past(inflow, step - pool.tau - pool.tau_bar)
            - pool.c[0] * past(drawn, step - pool.tau_bar)
        )

    b1, b2, b3 = pool.b
    c1, c2, c3 = pool.c
    a1, a2 = pool.alpha
    tau = pool.tau
    return (
        b1 * past(inflow, step - tau)
        - b2 * past(inflow, step - tau - 1)
        + b3 * past(inflow, step - tau - 2)
        - c1 * past(drawn, step)
        + c2 * past(drawn, step - 1)
        - c3 * past(drawn, step - 2)
        + level(step)
        + a1 * (level(step) - 2.0 * level(step - 1) + level(step - 2))
        + a2 * (level(step) - level(step - 1))
    )


@dataclass(frozen=True)
class ClosedLoop:
    """Everything needed to run the network for a while.

    ``plant_pools`` are the third-order models the response is evaluated
    on; ``design_pools`` are the first-order models with the re-fitted
    delays that the controller was designed against. They are different
    objects on purpose - keeping the evaluation model and the design model
    apart is the whole reason an observer is needed at all.
    """

    plant_pools: tuple[PoolParams, ...]
    design_pools: tuple[PoolParams, ...]
    filter_spec: FilterSpec
    weights: LqWeights
    kalman_r1: float = 1.0
    kalman_r2: float = 100.0
    kalman_reading: str = "covariance"

    def __post_init__(self) -> None:
        if len(self.plant_pools) != len(self.design_pools):
            raise ValueError(
                f"the plant has {len(self.plant_pools)} pools and the design model "
                f"has {len(self.design_pools)}"
            )
        if any(p.order != 3 for p in self.plant_pools):
            raise ValueError("the plant is evaluated on the third-order models")
        if any(p.order != 1 for p in self.design_pools):
            raise ValueError("the controller is designed on the first-order models")

    @property
    def n_pools(self) -> int:
        return len(self.plant_pools)

    @property
    def gain(self) -> float:
        return kalman_gain(self.kalman_r1, self.kalman_r2, self.kalman_reading)


@dataclass(frozen=True)
class ClosedLoopResult:
    """What a run produced, in the physical sign convention throughout.

    ``commanded`` is what the controller asked for, ``applied`` is what
    reached the plant after filtering, and they are different signals: the
    gap between them is the filter this study exists to work around.
    """

    levels: np.ndarray
    commanded: np.ndarray
    applied: np.ndarray
    offtake_applied: np.ndarray
    estimates: np.ndarray
    gain: float


def simulate_closed_loop(
    loop: ClosedLoop,
    initial_levels: np.ndarray,
    offtake: np.ndarray,
    horizon: int | None = None,
    law: LinearControlLaw | None = None,
) -> ClosedLoopResult:
    """Run the network with its controller for ``len(offtake)`` steps.

    Parameters
    ----------
    loop:
        The assembled network.
    initial_levels:
        Level deviation of each pool before the first step, most
        downstream first.
    offtake:
        Planned offtakes, shape ``(steps, n_pools)``, in the **physical**
        convention: a positive value is water leaving the canal. It is
        announced, so the controller sees all of it from the start; that
        is the source's assumption and it is what feed-forward acts on.
    horizon:
        Preview horizon of the control law. Defaults to the run length,
        which must be long enough for the law to stand in for the source's
        infinite-horizon one.
    law:
        A previously extracted law, to avoid recomputing it across runs
        that share a network.
    """
    offtake = np.asarray(offtake, dtype=float)
    initial_levels = np.asarray(initial_levels, dtype=float)
    n = loop.n_pools
    steps = offtake.shape[0]

    if offtake.ndim != 2 or offtake.shape[1] != n:
        raise ValueError(f"offtake must have shape (steps, {n}), got {offtake.shape}")
    if initial_levels.shape != (n,):
        raise ValueError(f"initial_levels must have shape ({n},)")

    if horizon is None:
        horizon = steps
    if law is None:
        law = control_law(list(loop.design_pools), loop.weights, horizon=horizon)
    if law.horizon != horizon:
        raise ValueError(
            f"the law was built for a horizon of {law.horizon}, not {horizon}"
        )

    numerator, denominator = loop.filter_spec.coefficients()
    order = max(len(numerator), len(denominator)) - 1
    input_filter_state = np.zeros((n, order))
    offtake_filter_state = np.zeros((n, order))

    levels = np.zeros((steps + 1, n))
    levels[0] = initial_levels
    commanded = np.zeros((steps, n))
    applied = np.zeros((steps, n))
    offtake_applied = np.zeros((steps, n))
    drawn = np.zeros((steps, n))
    estimates = np.zeros((steps + 1, n))
    estimates[0] = initial_levels  # the observer starts where the network does

    # The controller works in the source's disturbance convention.
    planned_source = -offtake

    gain = loop.gain
    lag = law.lag

    for t in range(steps):
        history = np.zeros((lag, n))
        for k in range(lag):
            if t - 1 - k >= 0:
                history[k] = commanded[t - 1 - k]

        preview = np.zeros((horizon, n))
        available = min(horizon, steps - t)
        if available > 0:
            preview[:available] = planned_source[t : t + available]

        announced = np.zeros((lag, n))
        for k in range(lag):
            if t - 1 - k >= 0:
                announced[k] = planned_source[t - 1 - k]

        # 1. command, from the a priori estimate
        command = law.input_at(estimates[t], history, preview, announced)
        commanded[t] = command

        # 2. filter the command and the planned offtake, channel by channel
        for i in range(n):
            value, input_filter_state[i] = signal.lfilter(
                numerator, denominator, [command[i]], zi=input_filter_state[i]
            )
            applied[t, i] = value[0]
            value, offtake_filter_state[i] = signal.lfilter(
                numerator, denominator, [offtake[t, i]], zi=offtake_filter_state[i]
            )
            offtake_applied[t, i] = value[0]

        # 3. drive the plant: everything leaving at the downstream end
        for i in range(n):
            outflow = applied[t, i - 1] if i >= 1 else 0.0
            drawn[t, i] = outflow + offtake_applied[t, i]
        for i, pool in enumerate(loop.plant_pools):
            levels[t + 1, i] = step_third_order(
                pool, levels[:, i], applied[:, i], drawn[:, i], t
            )

        # 4. correct on the measurement, then predict the next estimate
        corrected = estimates[t] + gain * (levels[t] - estimates[t])
        for i, pool in enumerate(loop.design_pools):
            inflow_lag = t - pool.tau - pool.tau_bar
            outflow_lag = t - pool.tau_bar
            outflow = (
                commanded[outflow_lag, i - 1] if (i >= 1 and outflow_lag >= 0) else 0.0
            )
            disturbance = (
                planned_source[outflow_lag, i] if outflow_lag >= 0 else 0.0
            )
            estimates[t + 1, i] = (
                corrected[i]
                + pool.b[0] * (commanded[inflow_lag, i] if inflow_lag >= 0 else 0.0)
                - pool.c[0] * (outflow - disturbance)
            )

    return ClosedLoopResult(
        levels=levels,
        commanded=commanded,
        applied=applied,
        offtake_applied=offtake_applied,
        estimates=estimates,
        gain=gain,
    )
