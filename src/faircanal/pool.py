"""Pool dynamics: the identified input-output model of one canal reach.

The model is the discrete-time pool model of Heyden, Pates and Rantzer
(arXiv:2203.16575), taken in its published form. Nothing about the plant,
the controller or the filter is changed anywhere in this study; the reach
model is reproduced here so that the closed loop can be assembled and the
order can be reshaped in front of it.

Sign convention
---------------
The level deviation ``y`` is measured at the downstream end of the pool.

* ``inflow`` is the discharge entering the pool. It raises the level after
  a transport delay ``tau``, because the water has to travel the reach
  before it is seen by the sensor at the far end.
* ``outflow`` is the discharge leaving the pool through its downstream
  gate or gates. It lowers the level immediately: the gate sits where the
  level is measured, so there is no travel time.
* ``offtake`` is the discharge drawn by the users at that gate. A positive
  offtake **takes water out of the pool** and therefore lowers the level,
  exactly like the outflow. This follows the description of the
  disturbance in the source (Heyden et al., arXiv:2203.16575, Fig. 1),
  which states that the disturbance takes water out of the pool.

The sign of the offtake term is the one thing in this module that is not
mechanical transcription: the equation as it appears in the extracted text
carries the opposite sign, which would make a withdrawal *raise* the
level. The prose is unambiguous and the equation is not, so the prose
wins. ``test_offtake_lowers_level`` guards the choice and fails the whole
suite if it is ever reversed.

Tree topology
-------------
In a cascade a pool has exactly one downstream gate. In a tree it has
several, and the level equation sees their sum. Passing the summed
outflow of the child pools is the only change; a cascade is the case of a
single child, so the two agree by construction
(``test_cascade_is_a_tree_with_one_child``).

Writing the sum this way assumes the branch gates sit at one cross
section, so that the level of the parent pool does not depend on *which*
branch the water leaves through. That is an assumption, not a
measurement, and it is declared as such wherever a composed topology is
reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PoolParams", "simulate_level", "homogeneous_roots", "total_outflow"]


@dataclass(frozen=True)
class PoolParams:
    """Identified coefficients of one pool.

    Parameters
    ----------
    name:
        Label used in reports and error messages.
    b:
        Inflow coefficients. One entry for the first-order model, three for
        the third-order model.
    c:
        Outflow and offtake coefficients, same length as ``b``.
    alpha:
        Wave-dynamics coefficients of the third-order model. Ignored, and
        required to be zero, for the first-order model.
    tau:
        Transport delay of the inflow, in plant steps.
    tau_bar:
        Additional delay contributed by the low-pass filter, in plant
        steps. Only defined for the first-order approximation, where the
        source folds it into the delay; it is rejected for order three.

    The coefficients carry units of s/m^2: the level is in metres and the
    discharges are in cubic metres per second.
    """

    name: str
    b: tuple[float, ...]
    c: tuple[float, ...]
    alpha: tuple[float, float] = (0.0, 0.0)
    tau: int = 0
    tau_bar: int = 0
    units: str = "s/m^2"
    provenance: str = "observed"
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.b) not in (1, 3):
            raise ValueError(f"{self.name}: b must have 1 or 3 entries, got {len(self.b)}")
        if len(self.c) != len(self.b):
            raise ValueError(
                f"{self.name}: b and c must have the same length, "
                f"got {len(self.b)} and {len(self.c)}"
            )
        if len(self.alpha) != 2:
            raise ValueError(f"{self.name}: alpha must have exactly 2 entries")
        if self.tau < 0 or self.tau_bar < 0:
            raise ValueError(f"{self.name}: delays must be non-negative")
        if self.order == 1:
            if any(self.alpha):
                raise ValueError(
                    f"{self.name}: the first-order model has no wave coefficients, "
                    f"but alpha={self.alpha}"
                )
        elif self.tau_bar:
            raise ValueError(
                f"{self.name}: tau_bar is only defined for the first-order "
                f"approximation, where the source folds the filter delay into the "
                f"transport delay"
            )

    @property
    def order(self) -> int:
        """Model order: 1 or 3."""
        return len(self.b)


def total_outflow(branch_flows: "list[np.ndarray] | tuple[np.ndarray, ...]") -> np.ndarray:
    """Sum the discharges leaving a pool through its downstream gates.

    In a cascade there is one branch and this returns it unchanged, which
    is why the tree model and the cascade model are the same model. In a
    tree there are several, and the parent reach sees their sum - the
    A-tree assumption described in the module docstring.

    An empty list is a leaf pool, with nothing leaving downstream; the
    caller must say how many steps that is.
    """
    flows = [np.asarray(flow, dtype=float) for flow in branch_flows]
    if not flows:
        raise ValueError(
            "a leaf pool has no downstream branches; pass an explicit array of "
            "zeros of the required length instead of an empty list"
        )
    shapes = {flow.shape for flow in flows}
    if len(shapes) != 1:
        raise ValueError(f"branch flows must all have the same shape, got {shapes}")
    if flows[0].ndim != 1:
        raise ValueError("branch flows must be one-dimensional")

    total = flows[0].copy()
    for flow in flows[1:]:
        total += flow
    return total


def _lagged(signal: np.ndarray, index: int) -> float:
    """Value of *signal* at *index*, taking anything before the start as zero."""
    if index < 0:
        return 0.0
    return float(signal[index])


def simulate_level(
    params: PoolParams,
    inflow: np.ndarray,
    outflow: np.ndarray,
    offtake: np.ndarray,
    y_initial: float = 0.0,
) -> np.ndarray:
    """Simulate the level deviation of one pool.

    Parameters
    ----------
    params:
        Identified coefficients of the pool.
    inflow:
        Discharge entering the pool, one value per plant step, in m^3/s.
    outflow:
        Discharge leaving through the downstream gate. In a tree this is
        the sum over the child pools; see the module docstring.
    offtake:
        Discharge drawn by the users at that gate. Positive values take
        water out of the pool.
    y_initial:
        Level deviation before the first step, in metres. The two steps
        before the start are held at the same value, so a simulation that
        starts from rest starts from rest in the third-order model too.

    Returns
    -------
    numpy.ndarray
        Level deviation in metres, of length ``len(inflow) + 1``: the
        initial value followed by one value per plant step.
    """
    inflow = np.asarray(inflow, dtype=float)
    outflow = np.asarray(outflow, dtype=float)
    offtake = np.asarray(offtake, dtype=float)

    if not (inflow.shape == outflow.shape == offtake.shape):
        raise ValueError(
            f"{params.name}: inflow, outflow and offtake must have the same shape, "
            f"got {inflow.shape}, {outflow.shape}, {offtake.shape}"
        )
    if inflow.ndim != 1:
        raise ValueError(f"{params.name}: signals must be one-dimensional")

    steps = inflow.size
    y = np.empty(steps + 1, dtype=float)
    y[0] = float(y_initial)

    # Everything leaving the pool at its downstream end, in one signal.
    drawn = outflow + offtake

    if params.order == 1:
        b1 = params.b[0]
        c1 = params.c[0]
        delay_in = params.tau + params.tau_bar
        delay_out = params.tau_bar
        for k in range(steps):
            y[k + 1] = (
                y[k]
                + b1 * _lagged(inflow, k - delay_in)
                - c1 * _lagged(drawn, k - delay_out)
            )
        return y

    b1, b2, b3 = params.b
    c1, c2, c3 = params.c
    a1, a2 = params.alpha
    tau = params.tau

    def level(index: int) -> float:
        """Level at *index*, holding the initial value before the start."""
        if index < 0:
            return float(y_initial)
        return float(y[index])

    for k in range(steps):
        y[k + 1] = (
            b1 * _lagged(inflow, k - tau)
            - b2 * _lagged(inflow, k - tau - 1)
            + b3 * _lagged(inflow, k - tau - 2)
            - c1 * _lagged(drawn, k)
            + c2 * _lagged(drawn, k - 1)
            - c3 * _lagged(drawn, k - 2)
            + level(k)
            + a1 * (level(k) - 2.0 * level(k - 1) + level(k - 2))
            + a2 * (level(k) - level(k - 1))
        )

    return y


def homogeneous_roots(params: PoolParams) -> np.ndarray:
    """Roots of the characteristic polynomial of the unforced pool.

    Collecting the homogeneous part of the third-order model gives

        y[k+1] = (1 + a1 + a2) y[k] - (2 a1 + a2) y[k-1] + a1 y[k-2],

    that is

        z^3 - (1 + a1 + a2) z^2 + (2 a1 + a2) z - a1 = 0.

    A canal pool is an integrator carrying a damped wave, so exactly one
    root sits on the unit circle and the rest sit strictly inside it. A
    parameter set that does not have that shape has been transcribed
    wrongly, which is what ``test_pool_is_an_integrator_with_a_damped_wave``
    checks.

    Returns the roots sorted by decreasing modulus, then by real part, so
    the order does not depend on the numerical routine.
    """
    if params.order == 1:
        # y[k+1] = y[k]: a pure integrator, one root at z = 1.
        return np.array([1.0])

    a1, a2 = params.alpha
    coefficients = [1.0, -(1.0 + a1 + a2), (2.0 * a1 + a2), -a1]
    roots = np.roots(coefficients)
    order = np.lexsort((roots.real, -np.abs(roots)))
    return roots[order]
