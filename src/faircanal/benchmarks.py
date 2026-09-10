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
    "haughton_pool",
    "haughton_filter",
]

HAUGHTON_SOURCE = (
    "Heyden M, Pates R, Rantzer A. Structured controller synthesis for "
    "irrigation networks. arXiv:2203.16575, Table 1. Object: Haughton main "
    "channel, Australia, pools 9 and 10. Sample time one minute. The "
    "coefficients originate from Ooi SK, Krutzen M, Weyer E (2005), Control "
    "Engineering Practice 13(4):461-471, and are reproduced openly in the "
    "arXiv paper."
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
#: The source notes that the first-order inflow coefficient is scaled by a
#: factor of one and a half to match the DC gain. Whether the tabulated
#: value is the scaled one or still has to be scaled is not stated, so no
#: scaling is applied here and the published number is used as printed.
#: The question is carried as an open item and is settled when the source
#: baseline is reproduced, not by guessing.
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

#: Low-pass filter of the source study. The order is reported as three in
#: the paper and as four in the authors' released code; both are carried
#: until reproducing the baseline settles which one produces the published
#: response. Nothing else about the filter is in dispute.
HAUGHTON_FILTER = {
    "family": "butterworth",
    "cutoff_rad_per_s": 3.0e-3,
    "sample_time_s": 60.0,
    "order_in_paper": 3,
    "order_in_released_code": 4,
    "applies_to": "each input and each planned disturbance",
    "provenance": "observed, except the order, which the sources disagree on",
    "source": HAUGHTON_SOURCE,
}


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
