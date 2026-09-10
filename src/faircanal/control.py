"""The structured LQ controller of the source study.

This study keeps the existing controller untouched and changes only the
order that reaches it, so the controller has to be *the* controller of the
source study rather than something similar. Getting it right therefore
matters as much as anything else here, and it is harder than it looks: the
published algorithm has two places where the typesetting leaves the
formula ambiguous.

So the algorithm is not the authority here. The paper's Theorem 1 says
that the algorithm computes the minimiser of the control problem (4), and
that problem is stated without ambiguity. The optimisation is therefore
implemented as well, and it is the arbiter: where the transcribed
algorithm and the directly solved problem agree, both are confirmed; where
they disagree, the transcription is wrong and the disagreement says where.

Sign convention - read this before using the module
---------------------------------------------------
The source writes its pool dynamics as

    y[t+1] = y[t] + b u[t - tau - taubar] - c (u_prev[t - taubar] - d[t - taubar])

so in **the source's own convention a positive d raises the level**. That
is the opposite of a withdrawal, and it contradicts the same paper's prose
and figure caption, both of which say the offtake takes water out of the
pool. The published time responses settle which one the authors simulated:
the gate flows rise while the disturbance is active, which is what
compensating a withdrawal looks like.

This module reproduces the controller, so it works in the source's
convention throughout and calls that quantity ``d_source``. The rest of
the package works in the physical convention, where a positive offtake is
water leaving the canal. The conversion is a single sign and it lives in
one place, :func:`offtake_to_source_disturbance`, so that it can be tested
rather than remembered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faircanal.pool import PoolParams

__all__ = [
    "LqWeights",
    "DesignParameters",
    "offtake_to_source_disturbance",
    "design_parameters",
    "simulate_first_order",
    "solve_lq_trajectory",
    "LinearControlLaw",
    "control_law",
]


def offtake_to_source_disturbance(offtake: np.ndarray) -> np.ndarray:
    """Convert a physical offtake into the source study's disturbance sign.

    A positive offtake is water leaving the canal, which lowers the level.
    A positive ``d`` in the source's equations raises it. The two differ by
    a sign and by nothing else.
    """
    return -np.asarray(offtake, dtype=float)


@dataclass(frozen=True)
class LqWeights:
    """Weights of the cost the controller minimises.

    The source's cost is

        sum_t sum_i  q_i y_i[t]^2 + r_i u_i[t]^2 + rho_i (u_i[t+1] - u_i[t])^2

    and the structured controller exists only for a restricted choice of
    it. The paper states the restriction outright: "The structured
    controller can only be used when rho_i = 0 for all inputs and r_i = 0
    for all inputs except for i = N". Anything else is refused here rather
    than silently designed for, because the result would be a controller
    the source's theorem says nothing about.
    """

    q: float = 1.0
    r_reservoir: float = 0.3
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.q <= 0.0:
            raise ValueError(f"q must be positive, got {self.q}")
        if self.r_reservoir <= 0.0:
            raise ValueError(
                f"r for the reservoir gate must be positive, got {self.r_reservoir}; "
                f"with no penalty on that flow the problem is unbounded"
            )
        if self.rho != 0.0:
            raise ValueError(
                "the structured controller requires rho = 0 for every input; "
                f"got rho={self.rho}. Penalising gate movement puts the design "
                "outside what the source's theorem covers."
            )


@dataclass(frozen=True)
class DesignParameters:
    """Output of the source's parameter algorithm.

    Attributes
    ----------
    gamma:
        One value per pool, indexed from the most downstream pool.
    b_hat:
        The rescaled inflow coefficients of the same sweep.
    q_scaled:
        The rescaled level weights.
    r_scaled:
        The reservoir-gate weight after the final rescaling.
    x:
        The scalar the reservoir gate's law is built from.
    g:
        The discount that weights previewed disturbances.
    """

    gamma: np.ndarray
    b_hat: np.ndarray
    q_scaled: np.ndarray
    r_scaled: float
    x: float
    g: float


def design_parameters(pools: list[PoolParams], weights: LqWeights) -> DesignParameters:
    """Compute the controller parameters by the source's sweep.

    Transcribed from Algorithm 1, which is unambiguous as printed:

        send gamma_1 = q_1 and b_hat_1 = b_1 upstream
        for i = 2..N:
            b_hat_i = b_i / c_i * b_hat_{i-1}
            q_i     = c_i^2 / b_hat_{i-1}^2 * q_i
            gamma_i = gamma_{i-1} q_i / (gamma_{i-1} + q_i)
        r = r / b_hat_N^2
        X = -gamma_N / 2 + sqrt(gamma_N r + gamma_N^2 / 4)
        g = X / (X + gamma_N)

    ``pools`` is ordered from the most downstream pool to the most
    upstream one, matching the source's indexing where pool N sits next to
    the reservoir.
    """
    if not pools:
        raise ValueError("at least one pool is required")

    n = len(pools)
    b = np.array([p.b[0] for p in pools], dtype=float)
    c = np.array([p.c[0] for p in pools], dtype=float)
    if np.any(b <= 0.0) or np.any(c <= 0.0):
        raise ValueError("the source's derivation assumes b_i > 0 and c_i > 0")

    b_hat = np.empty(n)
    q_scaled = np.full(n, weights.q, dtype=float)
    gamma = np.empty(n)

    b_hat[0] = b[0]
    gamma[0] = q_scaled[0]

    for i in range(1, n):
        b_hat[i] = b[i] / c[i] * b_hat[i - 1]
        q_scaled[i] = c[i] ** 2 / b_hat[i - 1] ** 2 * q_scaled[i]
        gamma[i] = gamma[i - 1] * q_scaled[i] / (gamma[i - 1] + q_scaled[i])

    r_scaled = weights.r_reservoir / b_hat[n - 1] ** 2
    x = -gamma[n - 1] / 2.0 + np.sqrt(
        gamma[n - 1] * r_scaled + gamma[n - 1] ** 2 / 4.0
    )
    g = x / (x + gamma[n - 1])

    return DesignParameters(
        gamma=gamma,
        b_hat=b_hat,
        q_scaled=q_scaled,
        r_scaled=float(r_scaled),
        x=float(x),
        g=float(g),
    )


# ---------------------------------------------------------------------------
# The design model and the control problem it poses
# ---------------------------------------------------------------------------


def simulate_first_order(
    pools: list[PoolParams],
    u: np.ndarray,
    d_source: np.ndarray,
    y_initial: np.ndarray,
    u_history: np.ndarray | None = None,
    d_history: np.ndarray | None = None,
) -> np.ndarray:
    """Run the first-order design model the controller is built on.

    The dynamics are the source's equation (3), reproduced exactly,
    including the sign of ``d``:

        y_i[t+1] = y_i[t] + b_i u_i[t - tau_i - taubar]
                          - c_i (u_{i-1}[t - taubar] - d_i[t - taubar])

    Pool 1 is the most downstream one; the flow over its downstream gate is
    fixed, so ``u_0`` is identically zero.

    Parameters
    ----------
    pools:
        One entry per pool, most downstream first, first-order models with
        the delays the source used for synthesis.
    u:
        Shape ``(steps, n_pools)``. Column ``i`` is the inflow to pool
        ``i+1``.
    d_source:
        Shape ``(steps, n_pools)``, in the source's sign convention.
    y_initial:
        Level of each pool before the first step.
    u_history, d_history:
        What happened before the first step, shape ``(lag, n_pools)`` with
        row ``k`` holding step ``-(k+1)``. Both default to nothing, which
        is a network starting from rest. They matter because the controller
        is a function of the buffered past inputs as much as of the current
        levels: a law extracted with the history assumed empty is a law for
        one instant only.

    Returns
    -------
    numpy.ndarray
        Shape ``(steps + 1, n_pools)``.
    """
    u = np.asarray(u, dtype=float)
    d_source = np.asarray(d_source, dtype=float)
    y_initial = np.asarray(y_initial, dtype=float)
    n = len(pools)

    if u.shape != d_source.shape:
        raise ValueError(f"u and d must have the same shape, got {u.shape} and {d_source.shape}")
    if u.ndim != 2 or u.shape[1] != n:
        raise ValueError(f"u must have shape (steps, {n}), got {u.shape}")
    if y_initial.shape != (n,):
        raise ValueError(f"y_initial must have shape ({n},), got {y_initial.shape}")
    if any(p.order != 1 for p in pools):
        raise ValueError("the design model is first order; pass first-order pools")
    for name, history in (("u_history", u_history), ("d_history", d_history)):
        if history is not None and (history.ndim != 2 or history.shape[1] != n):
            raise ValueError(f"{name} must have shape (lag, {n}), got {history.shape}")

    steps = u.shape[0]
    y = np.empty((steps + 1, n), dtype=float)
    y[0] = y_initial

    def sample(
        signal: np.ndarray,
        history: np.ndarray | None,
        step: int,
        column: int,
    ) -> float:
        """Value of *signal* at *step*, falling back to *history* before zero.

        The signal and its history are passed together rather than inferred
        from which array was handed in. An earlier version told them apart
        by object identity, which quietly broke the moment a caller passed
        the same zero array as both the input and the disturbance - the
        disturbance term then read the input history. Nothing raised, the
        model stayed linear, and the extracted controller was wrong.
        """
        if step >= 0:
            return float(signal[step, column])
        if history is None:
            return 0.0
        index = -step - 1
        if index >= history.shape[0]:
            return 0.0
        return float(history[index, column])

    for t in range(steps):
        for i, pool in enumerate(pools):
            inflow_lag = t - pool.tau - pool.tau_bar
            outflow_lag = t - pool.tau_bar
            # Pool 1's downstream gate is held fixed, so its outflow is zero.
            outflow = (
                sample(u, u_history, outflow_lag, i - 1) if i >= 1 else 0.0
            )
            y[t + 1, i] = (
                y[t, i]
                + pool.b[0] * sample(u, u_history, inflow_lag, i)
                - pool.c[0]
                * (outflow - sample(d_source, d_history, outflow_lag, i))
            )
    return y


def _impulse_maps(
    pools: list[PoolParams], steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Unit-pulse responses of the design model, one per input channel.

    The model is time invariant, so the response to a pulse at any step is
    the response to a pulse at step zero, shifted. Building the response
    maps from these is ``n`` simulations instead of ``steps * n``, which is
    the difference between a study that runs in seconds and one that runs
    in minutes.

    Returns two arrays of shape ``(n, steps + 1, n)``: the level response
    to a unit inflow pulse, and to a unit disturbance pulse.
    """
    n = len(pools)
    zeros = np.zeros((steps, n))
    h_u = np.empty((n, steps + 1, n))
    h_d = np.empty((n, steps + 1, n))
    for i in range(n):
        pulse = np.zeros((steps, n))
        pulse[0, i] = 1.0
        h_u[i] = simulate_first_order(pools, pulse, zeros, np.zeros(n))
        h_d[i] = simulate_first_order(pools, zeros, pulse, np.zeros(n))
    return h_u, h_d


def _toeplitz_from_impulse(h: np.ndarray, steps: int, n: int) -> np.ndarray:
    """Assemble a block lower-triangular response map from pulse responses."""
    a = np.zeros((steps * n, steps * n))
    for i in range(n):
        response = h[i]
        for t1 in range(steps):
            span = steps - t1
            block = response[1 : span + 1]  # y[t1+1 .. steps] from a pulse at t1
            a[t1 * n :, t1 * n + i] = block.ravel()
    return a


def _response_maps(
    pools: list[PoolParams], steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine maps giving the levels from the initial state, inputs and d.

    The design model is linear, so

        vec(y[1..steps]) = A0 @ y[0] + Au @ vec(u) + Ad @ vec(d).

    ``A0`` is built by simulating unit initial levels; ``Au`` and ``Ad`` are
    assembled from the pulse responses by time invariance.
    ``test_the_fast_response_maps_match_the_slow_ones`` checks the shortcut
    against building every column separately.
    """
    n = len(pools)
    zeros = np.zeros((steps, n))
    baseline = simulate_first_order(pools, zeros, zeros, np.zeros(n))[1:].ravel()
    assert not baseline.any(), "the design model is not at rest from rest"

    a0 = np.empty((steps * n, n))
    for i in range(n):
        y0 = np.zeros(n)
        y0[i] = 1.0
        a0[:, i] = simulate_first_order(pools, zeros, zeros, y0)[1:].ravel()

    h_u, h_d = _impulse_maps(pools, steps)
    return (
        a0,
        _toeplitz_from_impulse(h_u, steps, n),
        _toeplitz_from_impulse(h_d, steps, n),
    )


def _response_maps_by_column(
    pools: list[PoolParams], steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same maps, built one column at a time. Reference implementation.

    Kept only so the fast construction has something independent to be
    checked against; nothing in the study calls it.
    """
    n = len(pools)
    zeros = np.zeros((steps, n))
    rows = steps * n

    a0 = np.empty((rows, n))
    for i in range(n):
        y0 = np.zeros(n)
        y0[i] = 1.0
        a0[:, i] = simulate_first_order(pools, zeros, zeros, y0)[1:].ravel()

    au = np.empty((rows, rows))
    ad = np.empty((rows, rows))
    for column, (t, i) in enumerate((t, i) for t in range(steps) for i in range(n)):
        pulse = np.zeros((steps, n))
        pulse[t, i] = 1.0
        au[:, column] = simulate_first_order(pools, pulse, zeros, np.zeros(n))[1:].ravel()
        ad[:, column] = simulate_first_order(pools, zeros, pulse, np.zeros(n))[1:].ravel()
    return a0, au, ad


def solve_lq_trajectory(
    pools: list[PoolParams],
    weights: LqWeights,
    y_initial: np.ndarray,
    d_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the source's control problem (4) directly, over a finite horizon.

    This is the arbiter for the transcribed algorithm. The problem is

        minimise  sum_t sum_i q y_i[t]^2 + r u_N[t]^2
        subject to the design model, y[0] and d given,

    which is an unconstrained convex quadratic in the inputs, because the
    levels are affine in them. It is solved as a least-squares problem
    rather than by a Riccati recursion, so that nothing about the source's
    derivation is assumed while checking the source's derivation.

    The horizon is the length of ``d_source``. It must be long enough that
    the optimal input at the start has stopped depending on it;
    ``test_the_horizon_is_long_enough`` is what establishes that rather
    than a comment claiming it.

    Returns
    -------
    tuple
        The optimal inputs, the levels they produce, and the cost.
    """
    d_source = np.asarray(d_source, dtype=float)
    y_initial = np.asarray(y_initial, dtype=float)
    n = len(pools)
    steps = d_source.shape[0]

    a0, au, ad = _response_maps(pools, steps)
    offset = a0 @ y_initial + ad @ d_source.ravel()

    # Level cost: sqrt(q) * y for every pool and step.
    root_q = np.sqrt(weights.q)
    design = [root_q * au]
    target = [-root_q * offset]

    # Input cost: only the reservoir gate is penalised.
    root_r = np.sqrt(weights.r_reservoir)
    selector = np.zeros((steps, steps * n))
    for t in range(steps):
        selector[t, t * n + (n - 1)] = root_r
    design.append(selector)
    target.append(np.zeros(steps))

    solution, *_ = np.linalg.lstsq(
        np.vstack(design), np.concatenate(target), rcond=None
    )
    u = solution.reshape(steps, n)
    y = simulate_first_order(pools, u, d_source, y_initial)
    cost = weights.q * float(np.sum(y[1:] ** 2)) + weights.r_reservoir * float(
        np.sum(u[:, n - 1] ** 2)
    )
    return u, y, cost


# ---------------------------------------------------------------------------
# The control law, extracted from the problem rather than transcribed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearControlLaw:
    """The optimal control law of problem (4), as gains.

    The source's Theorem 1 says its algorithm computes the minimiser of
    problem (4). The minimiser of a linear-quadratic problem is a linear
    function of what the controller knows, so extracting that function from
    the problem gives the same law the algorithm implements - without
    depending on two lines of the published algorithm that are ambiguous as
    typeset.

    At each step,

        u[t] = k_levels @ y[t]
             + k_history @ vec(past inputs)
             + k_preview @ vec(disturbances from t onward)

    where the history rows run backwards from the previous step and the
    preview runs forward from the current one. ``k_preview`` is the
    feed-forward the source relies on instead of integral action, so a law
    with it dropped is a different controller, not a simplified one.
    """

    k_levels: np.ndarray
    k_history: np.ndarray
    k_preview: np.ndarray
    k_disturbance_history: np.ndarray
    lag: int
    horizon: int

    def input_at(
        self,
        levels: np.ndarray,
        history: np.ndarray,
        preview: np.ndarray,
        disturbance_history: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the law once.

        ``disturbance_history`` holds the offtakes announced before the
        current step that have not reached their pool yet. Omitting it
        assumes nothing is in that pipeline, which is true only at the
        start of a run.
        """
        n = self.k_levels.shape[0]
        if levels.shape != (n,):
            raise ValueError(f"levels must have shape ({n},), got {levels.shape}")
        if history.shape != (self.lag, n):
            raise ValueError(
                f"history must have shape ({self.lag}, {n}), got {history.shape}"
            )
        if preview.shape != (self.horizon, n):
            raise ValueError(
                f"preview must have shape ({self.horizon}, {n}), got {preview.shape}"
            )
        if disturbance_history is None:
            disturbance_history = np.zeros((self.lag, n))
        if disturbance_history.shape != (self.lag, n):
            raise ValueError(
                f"disturbance_history must have shape ({self.lag}, {n}), "
                f"got {disturbance_history.shape}"
            )
        return (
            self.k_levels @ levels
            + self.k_history @ history.ravel()
            + self.k_preview @ preview.ravel()
            + self.k_disturbance_history @ disturbance_history.ravel()
        )


def _least_squares_blocks(
    pools: list[PoolParams], weights: LqWeights, steps: int
) -> tuple[np.ndarray, float]:
    """The design matrix of problem (4) and the level weight's square root."""
    n = len(pools)
    _, au, _ = _response_maps(pools, steps)
    root_q = float(np.sqrt(weights.q))
    root_r = float(np.sqrt(weights.r_reservoir))

    selector = np.zeros((steps, steps * n))
    for step in range(steps):
        selector[step, step * n + (n - 1)] = root_r
    return np.vstack([root_q * au, selector]), root_q


def _history_map(
    pools: list[PoolParams], steps: int, lag: int, signal: str = "input"
) -> np.ndarray:
    """Levels at times 1..steps caused by inputs applied before time zero.

    This one cannot be assembled by shifting the impulse response, and the
    reason is worth stating because the shortcut looks obviously correct.

    An input applied at ``t = -(k+1)`` reaches different pools at different
    lags: the pool below the gate after its transport delay plus the filter
    delay, the pool above it after the filter delay alone. Part of its
    effect has therefore **already happened** by the time the window opens,
    and that part is sitting inside the level ``y[0]`` the caller supplies.
    Shifting the impulse response adds it a second time.

    So each column is simulated with the input placed in the history, which
    applies exactly the part that has not fired yet.
    ``test_the_history_map_does_not_double_count_the_past`` is what stops the
    shortcut coming back.

    ``signal`` selects which pipeline is being mapped. Announced offtakes
    have one too: a disturbance reaches a pool after the filter delay, so
    at any moment the ones announced within the last few steps are still on
    their way. A controller that sees only the future is blind to them, and
    the size of that blind spot is the delay itself.
    """
    if signal not in ("input", "disturbance"):
        raise ValueError(f"signal must be 'input' or 'disturbance', got {signal!r}")
    n = len(pools)
    no_input = np.zeros((steps, n))
    no_disturbance = np.zeros((steps, n))
    column_map = np.zeros((steps * n, lag * n))
    for i in range(n):
        for k in range(lag):
            history = np.zeros((lag, n))
            history[k, i] = 1.0
            blank = np.zeros((lag, n))
            column_map[:, k * n + i] = simulate_first_order(
                pools,
                no_input,
                no_disturbance,
                np.zeros(n),
                u_history=history if signal == "input" else blank,
                d_history=history if signal == "disturbance" else blank,
            )[1:].ravel()
    return column_map


def control_law(
    pools: list[PoolParams], weights: LqWeights, horizon: int, lag: int | None = None
) -> LinearControlLaw:
    """Extract the optimal control law of problem (4).

    One pseudo-inverse of the problem's design matrix gives the map from
    everything the controller knows to the whole optimal input trajectory;
    the first step of that trajectory is the law. Doing it this way means
    the law is optimal by construction rather than by transcription, and
    the horizon is the only approximation - it stands in for the source's
    infinite one, and how long it has to be was measured rather than
    assumed.

    ``lag`` defaults to the longest delay in the network, which is how far
    back the dynamics can still reach.
    """
    n = len(pools)
    if lag is None:
        lag = max(pool.tau + pool.tau_bar for pool in pools)
    if horizon < 1 or lag < 1:
        raise ValueError("horizon and lag must be positive")

    a0, _, ad = _response_maps(pools, horizon)
    design, root_q = _least_squares_blocks(pools, weights, horizon)
    ah = _history_map(pools, horizon, lag, "input")
    adh = _history_map(pools, horizon, lag, "disturbance")

    rows = horizon * n
    pseudo = np.linalg.pinv(design)
    transfer = -root_q * pseudo[:, :rows]

    return LinearControlLaw(
        k_levels=(transfer @ a0)[:n],
        k_history=(transfer @ ah)[:n],
        k_preview=(transfer @ ad)[:n],
        k_disturbance_history=(transfer @ adh)[:n],
        lag=lag,
        horizon=horizon,
    )
