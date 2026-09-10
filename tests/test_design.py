"""Permanent tests for the design models. Never deleted.

The controller is designed on these and run on something else, so an error
here does not break anything - it produces a controller tuned for a canal
that is not the one it runs on. The strongest check available is that the
source's own procedure, implemented here, returns the source's own
published integers; the rest of these tests are about the one thing the
source left unstated and this study had to work out.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from faircanal.benchmarks import (
    HAUGHTON_FILTER_DELAY,
    HAUGHTON_POOLS,
    haughton_design_pool,
    haughton_filter,
    haughton_pool,
)
from faircanal.delivery import memory_steps
from faircanal.design import (
    FIT_DRIVE_FACTOR,
    DesignError,
    crossover_steps,
    design_models,
    fit_drive,
    fit_filter_delay,
    fit_phases,
    fit_transport_delays,
)
from faircanal.network import corning_cascade, identify_network
from faircanal.pool import PoolParams, ramp_rate, simulate_level

SPEC = haughton_filter(3)
HAUGHTON = [haughton_pool(index, 3) for index in (1, 2)]
HAUGHTON_GAINS = tuple(
    (HAUGHTON_POOLS[1][index].b[0], HAUGHTON_POOLS[1][index].c[0]) for index in (1, 2)
)


def corning():
    cascade = corning_cascade()
    return list(identify_network(cascade))


def mismatch(plant, design, spec, phases) -> tuple[float, float]:
    """Worst relative gap between the design model and the filtered plant.

    Returned for the outflow channel and the inflow channel separately,
    because they carry different delays and a single number would hide
    which one is off.
    """
    steps = phases[2] + memory_steps(spec)
    drive = fit_drive(steps, phases)
    numerator, denominator = spec.coefficients()
    filtered = signal.lfilter(numerator, denominator, drive)
    rest = np.zeros(steps)
    worst_out = worst_in = 0.0
    for plant_model, design_model in zip(plant, design):
        reference = simulate_level(plant_model, rest, filtered, rest)
        candidate = simulate_level(design_model, rest, drive, rest)
        worst_out = max(
            worst_out, float(np.abs(candidate - reference).max() / np.abs(reference).max())
        )
        reference = simulate_level(plant_model, filtered, rest, rest)
        candidate = simulate_level(design_model, drive, rest, rest)
        worst_in = max(
            worst_in, float(np.abs(candidate - reference).max() / np.abs(reference).max())
        )
    return worst_out, worst_in


# ---------------------------------------------------------------------------
# The source's own numbers
# ---------------------------------------------------------------------------


def test_the_procedure_reproduces_the_published_delays():
    """Three integers the source printed, recomputed from its own procedure.

    The filter delay and the first pool's transport delay come out exactly.
    The second pool's lands one step away - sixteen against a published
    fifteen - on a residual where fifteen is about fifteen per cent worse.
    The source prints neither the drive's timings nor its horizon, and
    those are what decide between two neighbouring integers, so this is
    where a faithful reproduction is expected to land.
    """
    models, fit = design_models(HAUGHTON, SPEC, gains=HAUGHTON_GAINS)
    assert fit.filter_delay == HAUGHTON_FILTER_DELAY == 10
    assert fit.transport_delays[0] == 2
    assert fit.transport_delays[1] in (15, 16)
    assert models[0].tau == 2 and models[0].tau_bar == 10
    assert all(model.order == 1 for model in models)


def test_the_drive_length_rule_recovers_the_sources_own_drive():
    """Where the factor of eight came from, and that it is not a fudge.

    The source shows its fitting experiment over 0 to 150 minutes in three
    phases and never says why. Its two pools have a crossover of 6.26
    steps, and eight times that is fifty. So the one number this module
    chooses is a number the source's own parameters already contain.
    """
    crossover = crossover_steps(HAUGHTON)
    assert crossover == pytest.approx(6.26, abs=0.02)
    assert fit_phases(HAUGHTON) == (50, 100, 150)
    assert FIT_DRIVE_FACTOR == 8


def test_the_published_design_models_are_left_alone():
    """The Haughton set stays observed; this module never overwrites it."""
    assert haughton_design_pool(1).tau == 2
    assert haughton_design_pool(2).tau == 15
    assert haughton_design_pool(1).tau_bar == 10
    assert haughton_design_pool(1).provenance == "observed"


# ---------------------------------------------------------------------------
# The thing the source left unstated
# ---------------------------------------------------------------------------


def test_the_two_canals_tilt_by_very_different_amounts():
    """Why one drive length cannot serve both canals.

    A pool answers a flow change twice: at once, as the backwater surface
    tilts, and then slowly, as it fills. The ratio of the two is 6.3 steps
    on the source's small steep canal and 20.4 on the large flat one,
    because a seven-kilometre pool held under backwater tilts far more than
    it fills in a minute. Any drive shorter than that ratio is a drive the
    first-order model cannot fit at all.
    """
    assert crossover_steps(HAUGHTON) == pytest.approx(6.26, abs=0.05)
    assert crossover_steps(corning()) == pytest.approx(20.4, abs=0.1)
    ratios = [model.c[0] / ramp_rate(model)[1] for model in corning()]
    assert min(ratios) > 8.0
    assert max(ratios) > 19.0


def test_taking_the_printed_timings_literally_gets_it_wrong():
    """The mistake this rule exists to prevent, with its cost measured.

    Fitting the flat canal on the source's printed 50/100/150 leaves the
    design model twenty per cent from the plant it stands in for. With the
    drive scaled to that canal it is seven. The published timings are not
    wrong; they are the source's canal's timings, and carrying them across
    is the error.
    """
    plant = corning()
    scaled, _ = design_models(plant, SPEC)
    literal, _ = design_models(plant, SPEC, phases=(50, 100, 150))

    out_scaled, in_scaled = mismatch(plant, scaled, SPEC, fit_phases(plant))
    out_literal, in_literal = mismatch(plant, literal, SPEC, (50, 100, 150))

    assert out_scaled < 0.09, f"scaled drive now leaves {out_scaled:.1%}"
    assert in_scaled < 0.05
    assert out_literal > 0.15, f"the literal drive now leaves only {out_literal:.1%}"
    assert out_literal > 2.0 * out_scaled


def test_the_fit_does_not_depend_on_the_drive_length():
    """Between six and twenty crossovers the answer is the same, or nearly.

    A rule with one chosen number in it has to be shown not to depend on
    it. The filter delay is identical across the range and the transport
    delays move by at most one step.
    """
    plant = corning()
    answers = []
    for factor in (6, 8, 10, 15, 20):
        _, fit = design_models(plant, SPEC, phases=fit_phases(plant, factor))
        answers.append((fit.filter_delay, fit.transport_delays))
    filter_delays = {answer[0] for answer in answers}
    assert len(filter_delays) == 1, f"the filter delay moved: {filter_delays}"
    reference = answers[0][1]
    for _, delays in answers[1:]:
        assert all(abs(a - b) <= 1 for a, b in zip(reference, delays))


def test_the_fit_does_not_depend_on_which_half_comes_first():
    """Empty then fill, or fill then empty - the same delay either way."""
    for pools, gains in ((HAUGHTON, HAUGHTON_GAINS), (corning(), None)):
        phases = fit_phases(pools)
        pool_gains = gains or tuple(ramp_rate(model) for model in pools)
        forward, _ = fit_filter_delay(pools, pool_gains, SPEC, phases=phases, sign=1.0)
        backward, _ = fit_filter_delay(pools, pool_gains, SPEC, phases=phases, sign=-1.0)
        assert forward == backward


def test_the_flat_canal_needs_no_extra_delay_and_why():
    """A result that looks wrong until the two delays are counted.

    The fit gives ``tau_bar = 0`` on the flat canal, and confidently. That
    is not the filter being ignored: the filter delays a slow signal by
    about eleven steps and the pool's immediate backwater response arrives
    about twenty steps earlier than an integrator would have it, so on the
    outflow channel the two nearly cancel. The inflow channel keeps the
    difference instead, and its fitted delays come out above the pool's own.
    """
    plant = corning()
    models, fit = design_models(plant, SPEC)
    assert fit.filter_delay == 0
    assert fit.filter_margin > 0.15, (
        "the next integer is now close, so this is no longer a clear answer"
    )
    later = [
        design.tau - plant_model.tau for design, plant_model in zip(models, plant)
    ]
    assert min(later) >= -1, later
    assert max(later) >= 5, later
    assert sum(later) > 0, "the inflow channel is no longer absorbing the filter"

    # And the filter really does delay a slow signal by about eleven steps.
    numerator, denominator = SPEC.coefficients()
    frequencies, group = signal.group_delay((numerator, denominator), w=[1e-6, 1e-5])
    assert float(group[0]) == pytest.approx(11.08, abs=0.05)


def test_the_design_model_stands_in_for_the_filtered_plant():
    """The premise the whole design rests on, as a number for this canal.

    The source shows it as a figure and claims the first-order model
    captures the filtered third-order one well. Here it is measured on both
    canals: under ten per cent on the source's own, under eight on ours.
    """
    out_h, in_h = mismatch(
        HAUGHTON, design_models(HAUGHTON, SPEC, gains=HAUGHTON_GAINS)[0], SPEC,
        fit_phases(HAUGHTON),
    )
    assert out_h < 0.10 and in_h < 0.10

    plant = corning()
    out_c, in_c = mismatch(plant, design_models(plant, SPEC)[0], SPEC, fit_phases(plant))
    assert out_c < 0.09 and in_c < 0.05


# ---------------------------------------------------------------------------
# Shape of the answer
# ---------------------------------------------------------------------------


def test_the_gains_default_to_the_ramp_rate():
    """Nothing is fitted that is already known.

    The level change per step under a sustained flow is the one quantity
    the first-order and third-order models must agree on, and the
    construction of the third-order model makes it exact. So it is carried
    over rather than searched for, and only the delays are fitted.
    """
    plant = corning()
    models, _ = design_models(plant, SPEC)
    for design, plant_model in zip(models, plant):
        inflow, outflow = ramp_rate(plant_model)
        assert design.b[0] == pytest.approx(inflow, rel=1e-15)
        assert design.c[0] == pytest.approx(outflow, rel=1e-15)
        assert design.provenance == "derived"
        assert "Eq. (6)" in design.source


def test_a_delay_at_the_edge_of_the_search_is_refused():
    """A minimum at the boundary is not a minimum, and says so."""
    # The source's canal wants ten, so a search that stops at five ends on
    # its own boundary and has found nothing.
    with pytest.raises(DesignError, match="edge of the search"):
        fit_filter_delay(HAUGHTON, HAUGHTON_GAINS, SPEC, max_delay=5)

    plant = corning()
    gains = [ramp_rate(model) for model in plant]
    with pytest.raises(DesignError, match="edge of the search"):
        fit_transport_delays(
            plant, gains, SPEC, 0, phases=fit_phases(plant), max_delay=3
        )


def test_mismatched_inputs_are_refused():
    with pytest.raises(DesignError, match="gain pairs"):
        fit_filter_delay(HAUGHTON, HAUGHTON_GAINS[:1], SPEC)
    with pytest.raises(DesignError, match="do not fit inside"):
        fit_drive(40, (50, 100, 150))


def test_a_pool_that_does_not_ramp_is_refused():
    flat = PoolParams(
        name="flat", b=(0.1, 0.1, 0.0), c=(0.1, 0.1, 0.0), alpha=(0.0, 0.0), tau=1
    )
    assert ramp_rate(flat)[1] == pytest.approx(0.0)
    with pytest.raises(DesignError, match="no ramp"):
        crossover_steps([flat])


def test_the_fit_reports_how_close_the_runner_up_was():
    """An integer presented without its margin is an integer over-claimed."""
    _, fit = design_models(HAUGHTON, SPEC, gains=HAUGHTON_GAINS)
    assert 0.0 < fit.filter_margin < 1.0
    assert len(fit.margins) == 2
    assert all(margin > 0.0 for margin in fit.margins)
    assert fit.phases == (50, 100, 150)
    assert fit.steps == 150 + memory_steps(SPEC)
