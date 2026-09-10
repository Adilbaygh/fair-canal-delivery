"""The low-pass filter and the delivery operator.

The canal controller suppresses wave dynamics with a low-pass filter, and
the source study applies that filter to each input **and to each planned
disturbance**. A farmer therefore does not receive the discharge profile
that was ordered; the farmer receives its filtered image. That is the
mechanism this study is about, so the filter is reproduced here exactly as
published and is never modified.

What the filter costs the user
------------------------------
The filter has unit DC gain, so over an unbounded horizon it moves no
water at all: every cubic metre released is eventually delivered. Inside a
finite delivery window it is a different story. The impulse response has an
infinite tail, so part of any order arrives after the window closes, and
the response overshoots, so a rectangular order produces a filtered
profile that dips **below zero**. A negative discharge through an offtake
is not something a gate can do, which is why non-negativity of the
delivered flow is a constraint of the optimisation rather than a property
that comes for free.

Both facts are measured here rather than assumed: see
``dc_gain``, ``memory_steps`` and the tests that pin them.

The delivery operator
---------------------
Decisions are taken on blocks of ``steps_per_block`` plant steps and held
constant inside a block. Writing ``Z`` for that hold and ``G`` for the
filter as a lower-triangular Toeplitz matrix, the delivered discharge of
one user is

    q = G Z v = Gamma v,

which is linear in the reshaped order ``v``. That linearity is what makes
the whole problem a sequence of linear programmes, so ``Gamma`` is built
explicitly and checked against ``G @ Z`` rather than trusted.

Filter order
------------
The source paper reports a third-order filter and the authors' released
code uses a fourth-order one. Both are carried here, side by side, until
reproducing the published baseline settles which one produced it. Neither
is treated as the default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

__all__ = [
    "FilterSpec",
    "apply_filter",
    "impulse_response",
    "dc_gain",
    "memory_steps",
    "hold_matrix",
    "filter_matrix",
    "delivery_operator",
]


@dataclass(frozen=True)
class FilterSpec:
    """The low-pass filter of the source study.

    Parameters
    ----------
    order:
        Filter order. Three in the paper, four in the released code.
    cutoff_rad_per_s:
        Cut-off frequency in rad/s.
    sample_time_s:
        Plant sample time in seconds.
    family:
        Filter family. Only ``"butterworth"`` is published, and anything
        else is refused rather than silently substituted.
    """

    order: int
    cutoff_rad_per_s: float
    sample_time_s: float
    family: str = "butterworth"

    def __post_init__(self) -> None:
        if self.family != "butterworth":
            raise ValueError(
                f"the source specifies a Butterworth filter; refusing to "
                f"substitute {self.family!r}"
            )
        if self.order < 1:
            raise ValueError(f"filter order must be at least 1, got {self.order}")
        if self.cutoff_rad_per_s <= 0.0 or self.sample_time_s <= 0.0:
            raise ValueError("cut-off and sample time must be positive")
        if not 0.0 < self.normalised_cutoff < 1.0:
            raise ValueError(
                f"the cut-off must lie strictly below the Nyquist frequency; "
                f"normalised cut-off is {self.normalised_cutoff}"
            )

    @property
    def normalised_cutoff(self) -> float:
        """Cut-off as a fraction of the Nyquist frequency.

        The Nyquist frequency of a sampler with step ``T`` is ``pi / T``
        rad/s, so the normalisation is ``cutoff * T / pi``. Getting this
        wrong by a factor of two changes every result and nothing else
        would complain, so it is a property with a test rather than an
        expression inlined at the call site.
        """
        return self.cutoff_rad_per_s * self.sample_time_s / np.pi

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerator and denominator coefficients of the discrete filter."""
        return signal.butter(
            self.order, self.normalised_cutoff, btype="low", analog=False
        )


def apply_filter(spec: FilterSpec, x: np.ndarray) -> np.ndarray:
    """Filter *x* forward in time, starting from rest.

    Forward filtering only: the filter is part of the plant, so it is
    causal. A zero-phase filter would let the delivered flow anticipate
    the order, which is precisely the effect this study measures.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("the signal must be one-dimensional")
    b, a = spec.coefficients()
    return signal.lfilter(b, a, x)


def impulse_response(spec: FilterSpec, steps: int) -> np.ndarray:
    """First *steps* samples of the filter impulse response."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    impulse = np.zeros(steps, dtype=float)
    impulse[0] = 1.0
    return apply_filter(spec, impulse)


def dc_gain(spec: FilterSpec) -> float:
    """Steady-state gain of the filter, computed from its coefficients.

    Evaluating the transfer function at ``z = 1`` gives ``sum(b)/sum(a)``.
    This is the analytic value, not a truncated sum of the impulse
    response, so it does not depend on how long a horizon somebody chose.
    """
    b, a = spec.coefficients()
    return float(np.sum(b) / np.sum(a))


def memory_steps(
    spec: FilterSpec, threshold: float = 1.0e-4, max_steps: int = 100_000
) -> int:
    """How long the filter remembers, in plant steps.

    Returns the number of steps after which the impulse response stays
    below *threshold* in absolute value for good. The horizon of every
    experiment has to exceed the last delivery window by at least this
    much, or the filter tail falls off the end of the horizon and the
    volume balance silently stops adding up.

    Raises
    ------
    RuntimeError
        If the response has not settled within *max_steps*, rather than
        returning a number that means "we stopped looking".
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    response = np.abs(impulse_response(spec, max_steps))
    above = np.flatnonzero(response > threshold)
    if above.size == 0:
        return 0
    last = int(above[-1])
    if last >= max_steps - 1:
        raise RuntimeError(
            f"the impulse response had not settled below {threshold} within "
            f"{max_steps} steps; raise max_steps or check the filter spec"
        )
    return last + 1


def hold_matrix(steps: int, blocks: int, steps_per_block: int) -> np.ndarray:
    """Zero-order hold from decision blocks to plant steps.

    ``Z[k, j]`` is one when plant step ``k`` belongs to block ``j``. Steps
    beyond the last block get zero: a horizon longer than the decision
    window is exactly how the filter tail is given room to arrive.
    """
    if steps < 1 or blocks < 1 or steps_per_block < 1:
        raise ValueError("steps, blocks and steps_per_block must all be positive")
    if blocks * steps_per_block > steps:
        raise ValueError(
            f"the decision window ({blocks} blocks x {steps_per_block} steps = "
            f"{blocks * steps_per_block}) does not fit in a horizon of {steps} steps"
        )
    Z = np.zeros((steps, blocks), dtype=float)
    for j in range(blocks):
        Z[j * steps_per_block : (j + 1) * steps_per_block, j] = 1.0
    return Z


def filter_matrix(spec: FilterSpec, steps: int) -> np.ndarray:
    """The filter as a lower-triangular Toeplitz matrix over *steps* steps.

    Built for verification and for small problems. The delivery operator
    does not go through it, because it is quadratic in the horizon while
    the operator itself is not.
    """
    h = impulse_response(spec, steps)
    G = np.zeros((steps, steps), dtype=float)
    for k in range(steps):
        G[k, : k + 1] = h[k::-1]
    return G


def delivery_operator(
    spec: FilterSpec, steps: int, blocks: int, steps_per_block: int
) -> np.ndarray:
    """The operator ``Gamma = G Z`` mapping a block order to delivered flow.

    Column ``j`` is the filtered response to a unit order held over block
    ``j`` alone, so the operator is assembled with ``blocks`` filter runs
    rather than by multiplying two matrices of the horizon's size.
    ``test_delivery_operator_equals_g_times_z`` checks that the two agree.
    """
    Z = hold_matrix(steps, blocks, steps_per_block)
    gamma = np.empty((steps, blocks), dtype=float)
    for j in range(blocks):
        gamma[:, j] = apply_filter(spec, Z[:, j])
    return gamma
