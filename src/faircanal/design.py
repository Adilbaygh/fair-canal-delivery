"""The first-order models the controller is designed on, fitted not copied.

Why this module exists
----------------------
The controller of the source study is designed on a first-order model of
each pool and then run on the third-order one. That first-order model is
not the third-order model with terms dropped: it carries an extra delay
that stands in for the low-pass filter, and a transport delay that was
re-fitted rather than taken from the identification. For the source's own
canal those three integers are printed in the paper. For any other canal
they are not, and this study runs on another canal.

So the procedure is implemented rather than the numbers copied. The source
states it in full, in one paragraph:

    "First the optimal tau-bar is found by simulating the response for both
    pools to an outflow u_{i-1} corresponding to a constant positive input,
    followed by a constant zero input, followed by a negative input. ...
    The idea is that this describes when a pool is emptied and then filled.
    ... The value for tau-bar that minimizes the least square error
    (normalized for each pool) is chosen. ... Next, the optimal tau_i for
    each pool is found by minimizing the least square error when the
    inflow u_i is as in (6)."

Two delays, fitted in that order, and the order matters: in the first-order
model the outflow carries ``tau_bar`` alone while the inflow carries
``tau + tau_bar``, so driving the outflow isolates the filter's delay and
driving the inflow then leaves only the transport delay to find.

How long the drive has to be, and where that came from
------------------------------------------------------
The source prints the drive's shape and not its timings, and the timings
turn out to matter - not by a little. A pool answers a flow change twice:
at once, by the backwater surface tilting, and then slowly, by filling.
The first-order model has only the second of those, so a drive short
enough for the first to dominate is a drive the model cannot fit at all.

The crossover between the two is a time, and it is a ratio of numbers the
model already carries: the immediate response ``c1`` divided by the level
change per step under a sustained flow. :func:`crossover_steps` returns
it. On the source's own canal it is 6.3 steps; on the flat canal this
study runs on it is 20.4, because a seven-kilometre pool held under
backwater tilts far more than it fills in a minute.

The drive's phase is set to :data:`FIT_DRIVE_FACTOR` times that crossover.
The factor is eight, and it is eight because on the source's own pools
that is what their printed drive is: 6.3 steps times eight is 50, and the
figure they print runs 0 to 150 in three phases of 50. An undocumented
choice recovered from the paper's own parameters is the best evidence
available for it, and the tests show the answer is stable between six
times and twenty.

What the reproduction found
---------------------------
Run on the source's own two pools with the drive that rule gives, the
procedure returns tau-bar = 10 and tau_1 = 2, both exactly as published,
and tau_2 = 16 against a published 15. The residual at 15 is about fifteen
per cent above the residual at 16, on a residual that is itself small, so
this is where a reproduction is expected to land rather than a
disagreement. ``test_the_procedure_reproduces_the_published_delays``
records all three.

The Haughton design models keep their published delays regardless: they
are ``observed`` and this module never overwrites them. The fit is used
for the canal the experiments run on, where nothing is published to copy.

What it found on the other canal
--------------------------------
On the flat canal the fit returns ``tau_bar = 0``, and confidently - the
next integer is thirty per cent worse. That is not a failure and it is not
the filter being ignored. The filter delays the signal by about eleven
steps; the pool's immediate backwater response arrives about twenty steps
*earlier* than an integrator would have it. The two nearly cancel, and
what is left on the outflow channel is no net delay at all. The inflow
channel keeps the difference instead: the fitted transport delays come out
one to eight steps above the pool's own.

With a drive of the right length the first-order model then tracks the
filtered third-order one to under six per cent, which is better than the
ten per cent the same comparison gives on the source's own canal. With the
source's printed timings taken literally it would have been twenty per
cent, which is how the length rule came to be looked at at all.

Source
------
Heyden M, Pates R, Rantzer A (2022). A Structured Optimal Controller for
Irrigation Networks. ECC, doi:10.23919/ECC55457.2022.9838239; preprint
arXiv:2203.16575, Section 4.2 and Eq. (6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from faircanal.delivery import FilterSpec, memory_steps
from faircanal.pool import PoolParams, ramp_rate, simulate_level

__all__ = [
    "FIT_DRIVE_FACTOR",
    "DelayFit",
    "DesignError",
    "crossover_steps",
    "fit_phases",
    "fit_drive",
    "fit_filter_delay",
    "fit_transport_delays",
    "design_models",
]

#: How many crossover times one phase of the drive lasts. See the module
#: docstring: eight is what the source's own printed drive works out to on
#: the source's own pools.
FIT_DRIVE_FACTOR = 8


class DesignError(ValueError):
    """Raised when a design model cannot be fitted."""


@dataclass(frozen=True)
class DelayFit:
    """What the fit found, and how sure it was.

    ``filter_delay`` is one integer for the whole canal, as the source
    fits it. ``transport_delays`` is one per pool, in the same order the
    pools were given. ``margins`` is how much worse the next-best integer
    was, as a fraction - a small margin means the two neighbours are
    nearly as good and the answer should not be read as exact.
    """

    filter_delay: int
    transport_delays: tuple[int, ...]
    filter_margin: float
    margins: tuple[float, ...]
    phases: tuple[int, int, int]
    steps: int


def crossover_steps(
    models: "tuple[PoolParams, ...] | list[PoolParams]",
) -> float:
    """How long the immediate response outlasts the filling, in steps.

    A flow change tilts the backwater surface at once by ``c1`` and then
    fills the pool at the ramp rate. Their ratio is the time at which the
    filling has caught up with the tilt, and it is the shortest anything a
    first-order model is asked to fit can usefully be. The largest over the
    pools is returned, because one drive has to serve them all.
    """
    worst = 0.0
    for model in models:
        _, outflow = ramp_rate(model)
        if outflow <= 0.0:
            raise DesignError(f"{model.name}: a pool with no ramp cannot be fitted")
        worst = max(worst, abs(model.c[0]) / outflow)
    if worst <= 0.0:
        raise DesignError("no pool has an immediate response to fit against")
    return worst


def fit_phases(
    models: "tuple[PoolParams, ...] | list[PoolParams]",
    factor: float = FIT_DRIVE_FACTOR,
) -> tuple[int, int, int]:
    """Phase ends of the drive, scaled to the canal being fitted."""
    phase = max(1, int(round(factor * crossover_steps(models))))
    return phase, 2 * phase, 3 * phase


def fit_drive(
    steps: int, phases: "tuple[int, int, int]", sign: float = 1.0
) -> np.ndarray:
    """The drive of Eq. (6): draw, rest, fill, rest.

    A pool emptied and then filled, which is what makes the delay visible
    twice with opposite signs. ``sign`` swaps which half comes first; the
    fit is invariant to it, and the test says so.
    """
    first, second, third = phases
    if not 0 < first < second < third <= steps:
        raise DesignError(
            f"the drive's phases {phases} do not fit inside {steps} steps"
        )
    drive = np.zeros(steps)
    drive[:first] = sign
    drive[second:third] = -sign
    return drive


def _filtered(spec: FilterSpec, drive: np.ndarray) -> np.ndarray:
    numerator, denominator = spec.coefficients()
    return signal.lfilter(numerator, denominator, drive)


def _best_two(errors: "list[tuple[float, int]]") -> tuple[int, float]:
    """The winning delay and how much worse the runner-up was."""
    ordered = sorted(errors)
    best_error, best = ordered[0]
    if len(ordered) == 1 or best_error <= 0.0:
        return best, float("inf")
    return best, (ordered[1][0] - best_error) / best_error


def fit_filter_delay(
    plant_models: "tuple[PoolParams, ...] | list[PoolParams]",
    gains: "tuple[tuple[float, float], ...] | list[tuple[float, float]]",
    spec: FilterSpec,
    phases: "tuple[int, int, int] | None" = None,
    steps: int | None = None,
    max_delay: int | None = None,
    sign: float = 1.0,
) -> tuple[int, float]:
    """The one delay that stands in for the filter, for the whole canal.

    Every pool is driven on its **outflow**, where the first-order model
    carries ``tau_bar`` and nothing else, so the transport delays cannot
    contaminate the answer. The squared error of each pool is divided by
    that pool's own response before the pools are added up - the source's
    "normalized for each pool" - because otherwise the pool with the larger
    level swing would decide for both.
    """
    if len(plant_models) != len(gains):
        raise DesignError(f"{len(plant_models)} pools but {len(gains)} gain pairs")
    if phases is None:
        phases = fit_phases(plant_models)
    if steps is None:
        steps = phases[2] + memory_steps(spec)
    if max_delay is None:
        max_delay = memory_steps(spec)

    drive = fit_drive(steps, phases, sign)
    filtered = _filtered(spec, drive)
    rest = np.zeros(steps)

    references = []
    for model in plant_models:
        reference = simulate_level(model, rest, filtered, rest)
        scale = float(np.abs(reference).max())
        if scale <= 0.0:
            raise DesignError(f"{model.name}: the drive moves nothing")
        references.append((reference, scale))

    errors = []
    for delay in range(max_delay + 1):
        total = 0.0
        for (reference, scale), (inflow_gain, outflow_gain) in zip(references, gains):
            candidate = PoolParams(
                name="fit", b=(inflow_gain,), c=(outflow_gain,), tau=0, tau_bar=delay
            )
            level = simulate_level(candidate, rest, drive, rest)
            total += float(((level - reference) ** 2).sum()) / scale**2
        errors.append((total, delay))

    delay, margin = _best_two(errors)
    if delay == max_delay:
        raise DesignError(
            f"the filter delay came out at the edge of the search, {max_delay}; "
            f"widen it or the answer is a boundary, not a minimum"
        )
    return delay, margin


def fit_transport_delays(
    plant_models: "tuple[PoolParams, ...] | list[PoolParams]",
    gains: "tuple[tuple[float, float], ...] | list[tuple[float, float]]",
    spec: FilterSpec,
    filter_delay: int,
    phases: "tuple[int, int, int] | None" = None,
    steps: int | None = None,
    max_delay: int | None = None,
    sign: float = 1.0,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """One transport delay per pool, with the filter delay already fixed.

    Driven on the **inflow**, where the first-order model carries ``tau +
    tau_bar``. Since ``tau_bar`` is known by now, what is left to find is
    the transport delay alone.
    """
    if phases is None:
        phases = fit_phases(plant_models)
    if steps is None:
        steps = phases[2] + memory_steps(spec)
    if max_delay is None:
        max_delay = max(model.tau for model in plant_models) + memory_steps(spec)

    drive = fit_drive(steps, phases, sign)
    filtered = _filtered(spec, drive)
    rest = np.zeros(steps)

    delays: list[int] = []
    margins: list[float] = []
    for model, (inflow_gain, outflow_gain) in zip(plant_models, gains):
        reference = simulate_level(model, filtered, rest, rest)
        scale = float(np.abs(reference).max())
        if scale <= 0.0:
            raise DesignError(f"{model.name}: the drive moves nothing")
        errors = []
        for delay in range(max_delay + 1):
            candidate = PoolParams(
                name="fit",
                b=(inflow_gain,),
                c=(outflow_gain,),
                tau=delay,
                tau_bar=filter_delay,
            )
            level = simulate_level(candidate, drive, rest, rest)
            errors.append((float(((level - reference) ** 2).sum()) / scale**2, delay))
        delay, margin = _best_two(errors)
        if delay == max_delay:
            raise DesignError(
                f"{model.name}: the transport delay came out at the edge of the "
                f"search, {max_delay}"
            )
        delays.append(delay)
        margins.append(margin)
    return tuple(delays), tuple(margins)


def design_models(
    plant_models: "tuple[PoolParams, ...] | list[PoolParams]",
    spec: FilterSpec,
    phases: "tuple[int, int, int] | None" = None,
    steps: int | None = None,
    gains: "tuple[tuple[float, float], ...] | None" = None,
) -> tuple[tuple[PoolParams, ...], DelayFit]:
    """First-order design models for a whole canal, and the fit that made them.

    ``gains`` defaults to each pool's own ramp rate - the level change per
    step under a sustained unit flow, which is the quantity the first-order
    and third-order models must agree on and the one the source says it
    left unchanged. Pass it explicitly to reproduce a published first-order
    model whose gains were tabulated rather than derived.

    Returns the models in the order the pools were given, and the
    :class:`DelayFit` that produced them, so that a report can say how
    close the runner-up was rather than presenting an integer as exact.
    """
    if gains is None:
        gains = tuple(ramp_rate(model) for model in plant_models)
    if phases is None:
        phases = fit_phases(plant_models)
    filter_delay, filter_margin = fit_filter_delay(
        plant_models, gains, spec, phases, steps
    )
    delays, margins = fit_transport_delays(
        plant_models, gains, spec, filter_delay, phases, steps
    )
    models = tuple(
        PoolParams(
            name=f"{model.name}-design",
            b=(inflow_gain,),
            c=(outflow_gain,),
            tau=delay,
            tau_bar=filter_delay,
            units=model.units,
            provenance="derived",
            source=(
                "First-order design model fitted by the procedure of Heyden M, "
                "Pates R, Rantzer A (2022), doi:10.23919/ECC55457.2022.9838239, "
                "Section 4.2 and Eq. (6). Gains are the third-order ramp rate; "
                "delays are the least-squares fit against the filtered "
                "third-order response."
            ),
        )
        for model, delay, (inflow_gain, outflow_gain) in zip(
            plant_models, delays, gains
        )
    )
    fit = DelayFit(
        filter_delay=filter_delay,
        transport_delays=delays,
        filter_margin=filter_margin,
        margins=margins,
        phases=tuple(phases),
        steps=steps if steps is not None else phases[2] + memory_steps(spec),
    )
    return models, fit
