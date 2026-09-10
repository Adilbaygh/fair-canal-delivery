"""Published parameter sets, transcribed and kept apart.

Two independent parameter sets are used in this study and they are never
mixed:

* ``HAUGHTON`` - the identified pool models of the Haughton main channel,
  used only to reproduce the baseline of the source study in its own
  units;
* the ASCE test canal geometries, from which pool models are identified
  here rather than copied, used for the experiments.

Mixing them would produce numbers that belong to no canal. Every accessor
in this module names its set, and there is no function that returns both.

Provenance labels follow the convention used throughout the study:
``observed`` for a value read from a published source, ``derived`` for one
computed from observed values, ``assumed`` for one chosen by the authors.
"""

from __future__ import annotations

from faircanal.delivery import FilterSpec
from faircanal.pool import PoolParams

__all__ = [
    "HAUGHTON_SOURCE",
    "HAUGHTON_UNITS_NOTE",
    "HAUGHTON_POOLS",
    "HAUGHTON_FILTER",
    "HAUGHTON_DESIGN_DELAYS",
    "HAUGHTON_FILTER_DELAY",
    "HAUGHTON_SCENARIO",
    "haughton_pool",
    "haughton_filter",
    "haughton_design_pool",
]

HAUGHTON_SOURCE = (
    "Heyden M, Pates R, Rantzer A (2022). A Structured Optimal Controller for "
    "Irrigation Networks. European Control Conference (ECC), "
    "doi:10.23919/ECC55457.2022.9838239. Preprint: arXiv:2203.16575. "
    "Coefficients from Table 1, whose caption reads: 'Parameters for first and "
    "third-order models. For first-order: b_i = b_{i,1} and c_i = c_{i,1}. "
    "Sample time: one minute.' The two reaches are labelled 1 and 2 there and "
    "belong to the Haughton main channel, Australia."
)

HAUGHTON_UNITS_NOTE = (
    "The source declares no unit for u, y or the Table 1 coefficients, only "
    "a one-minute sample time. The implied effective surface area is far "
    "smaller than the geometric water surface of a reach this long, so the "
    "ratio of sample time to inflow coefficient is read as the effective "
    "storage area of the integrator-delay model rather than as a geometric "
    "area. This is why the Haughton set is used only to reproduce the "
    "source baseline in its own units, and never alongside a geometry-based "
    "parameter set."
)

#: Third-order models, Table 1, transcribed verbatim. Pool 1 is the more
#: upstream of the two reported reaches; the labels follow the source.
_HAUGHTON_ORDER_3 = {
    1: PoolParams(
        name="haughton-pool-1-order-3",
        b=(0.137, 0.155, 0.053),
        c=(0.190, 0.333, 0.175),
        alpha=(0.978, 0.468),
        tau=3,
        units="not declared in the source",
        provenance="observed",
        source=HAUGHTON_SOURCE,
    ),
    2: PoolParams(
        name="haughton-pool-2-order-3",
        b=(0.134, 0.244, 0.114),
        c=(0.101, 0.185, 0.087),
        alpha=(0.314, 0.814),
        tau=16,
        units="not declared in the source",
        provenance="observed",
        source=HAUGHTON_SOURCE,
    ),
}

#: First-order approximations, Table 1, transcribed verbatim.
#:
#: Section 4.2 says the first-order inflow coefficient was "increased by a
#: factor of 1.5" to match the third-order DC gain, and does not say
#: whether the tabulated numbers are before or after that change. They are
#: after: the tabulated value for the second pool matches the third-order
#: ramp rate to within one per cent, while dividing it by 1.5 would leave
#: it thirty-four per cent low and multiplying by 1.5 again would put it
#: forty-nine per cent high. So the published numbers are used exactly as
#: printed, and test_first_order_matches_the_third_order_ramp_rate keeps
#: anybody from applying the factor a second time.
_HAUGHTON_ORDER_1 = {
    1: PoolParams(
        name="haughton-pool-1-order-1",
        b=(0.069,),
        c=(0.063,),
        tau=3,
        units="not declared in the source",
        provenance="observed",
        source=HAUGHTON_SOURCE,
    ),
    2: PoolParams(
        name="haughton-pool-2-order-1",
        b=(0.0213,),
        c=(0.0156,),
        tau=14,
        units="not declared in the source",
        provenance="observed",
        source=HAUGHTON_SOURCE,
    ),
}

HAUGHTON_POOLS = {1: _HAUGHTON_ORDER_1, 3: _HAUGHTON_ORDER_3}

#: Low-pass filter of the source study.
#:
#: The order is settled by the paper itself, Section 4.1, verbatim: "A
#: Butterworth filter is used for the low-pass filter, as it has minimal
#: effect on the pass-band. The Matlab command butter is used for the
#: design and the final design is a third-order filter with a cut-off
#: frequency 3e-3 rad/sec." The authors' released code uses a fourth-order
#: filter; that is a difference between their paper and their code, not an
#: open question about the paper, and this study follows the paper. The
#: fourth order is kept reachable so the choice can be shown not to drive
#: the results.
#:
#: What the filter is applied to is settled in the same section: "We
#: instead choose to add a low-pass filter to each input and each planned
#: disturbance. Filtering the disturbance is natural since waves should be
#: avoided both in the pools and in the off-takes to the farmers." The
#: order a farmer places is therefore filtered before it is delivered.
HAUGHTON_FILTER = {
    "family": "butterworth",
    "cutoff_rad_per_s": 3.0e-3,
    "sample_time_s": 60.0,
    "order_in_paper": 3,
    "order_in_released_code": 4,
    "order_used": 3,
    "applies_to": "each input and each planned disturbance",
    "provenance": "observed",
    "source": HAUGHTON_SOURCE,
}

#: The first-order models the source study actually used for controller
#: synthesis, which are NOT the first-order rows of Table 1.
#:
#: Section 4.2, verbatim: "We also let tau_i be different from the ones in
#: [14], as it was noted that this had a positive effect on the
#: performance. The parameters b_i and c_i are unchanged." And: "The
#: resulting value for tau_bar is 10. ... The resulting value for the first
#: pool model is tau_i = 2 and for the second pool model tau_i = 15."
#:
#: Table 1 lists tau = 3 and tau = 14 for the first-order rows. Using those
#: for the controller reproduces a different study, so the two sets are
#: kept apart here the same way the Haughton and ASCE sets are.
HAUGHTON_DESIGN_DELAYS = {1: 2, 2: 15}
HAUGHTON_FILTER_DELAY = 10


def haughton_pool(index: int, order: int) -> PoolParams:
    """Return one Haughton pool model.

    Parameters
    ----------
    index:
        Pool label as used in the source: 1 or 2.
    order:
        Model order: 1 for the approximation, 3 for the full model.
    """
    if order not in HAUGHTON_POOLS:
        raise ValueError(f"order must be 1 or 3, got {order}")
    pools = HAUGHTON_POOLS[order]
    if index not in pools:
        raise ValueError(f"pool index must be one of {sorted(pools)}, got {index}")
    return pools[index]


def haughton_filter(order: int) -> FilterSpec:
    """Return the source study's low-pass filter at the requested order.

    The order is required rather than defaulted: the paper reports three
    and the released code uses four, and picking one silently would bury
    the disagreement instead of settling it.
    """
    candidates = (
        HAUGHTON_FILTER["order_in_paper"],
        HAUGHTON_FILTER["order_in_released_code"],
    )
    if order not in candidates:
        raise ValueError(
            f"the sources report orders {candidates}; refusing to build a "
            f"filter of order {order} without a reason recorded first"
        )
    return FilterSpec(
        order=order,
        cutoff_rad_per_s=HAUGHTON_FILTER["cutoff_rad_per_s"],
        sample_time_s=HAUGHTON_FILTER["sample_time_s"],
        family=HAUGHTON_FILTER["family"],
    )


#: The baseline scenario of Section 6, Figure 5.
#:
#: A five pool network alternating between the two identified reaches:
#: "Canal one, three, and five are modeled as the first pool and canal two
#: and four are modeled as the second pool." Indexing follows the source,
#: where pool N is the most upstream one, next to the reservoir.
#:
#: The initial condition is stated twice and the two statements disagree.
#: The caption of Figure 5 reads "[-5, 0, 0, 0, 5]" and the body text reads
#: "[5, 0, 0, 0, -5]". The caption is the one used here, because Figure 7
#: states the same kind of set-point change explicitly as "y_1 = -1, y_N =
#: 1", that is low downstream and high upstream, which is the caption's
#: ordering read as y_1 to y_N. The disagreement is recorded rather than
#: resolved silently, and reproducing the published response is what
#: confirms the reading.
HAUGHTON_SCENARIO = {
    "n_pools": 5,
    "pool_model_by_canal": (1, 2, 1, 2, 1),
    "initial_level_figure_caption": (-5.0, 0.0, 0.0, 0.0, 5.0),
    "initial_level_body_text": (5.0, 0.0, 0.0, 0.0, -5.0),
    "initial_level_used": (-5.0, 0.0, 0.0, 0.0, 5.0),
    "disturbance_pool": 1,
    "disturbance_start_step": 250,
    "disturbance_end_step": 450,
    "disturbance_level_change_per_step": 1.0,
    "cost_q": 1.0,
    "cost_r_reservoir_gate": 0.3,
    "cost_r_other_gates": 0.0,
    "cost_rho": 0.0,
    "kalman_r1": 1.0,
    "kalman_r2": 100.0,
    "provenance": "observed, except the initial condition, which the source states two ways",
    "source": HAUGHTON_SOURCE,
}


def haughton_design_pool(index: int) -> PoolParams:
    """Return the first-order model the source used for controller synthesis.

    Same b and c as the Table 1 first-order rows - Section 4.2 says "The
    parameters b_i and c_i are unchanged" - but with the re-fitted
    transport delay and the filter delay the paper reports, not the delays
    printed in Table 1.
    """
    table = haughton_pool(index, 1)
    return PoolParams(
        name=f"haughton-pool-{index}-design",
        b=table.b,
        c=table.c,
        tau=HAUGHTON_DESIGN_DELAYS[index],
        tau_bar=HAUGHTON_FILTER_DELAY,
        units=table.units,
        provenance="observed",
        source=(
            table.source
            + " Delays from Section 4.2: tau_bar = 10 for every pool, tau = 2 for "
            "the first pool model and tau = 15 for the second, re-fitted by the "
            "authors and different from the tau printed in Table 1."
        ),
    )
