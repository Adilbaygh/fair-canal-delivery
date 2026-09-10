"""Permanent tests for the frozen settings. Never deleted.

A silent change to a tolerance or a discretisation constant changes every
published number without changing a single line of the model, which is why
these are asserted rather than trusted.
"""

from __future__ import annotations

from faircanal import config
from faircanal.benchmarks import HAUGHTON_FILTER, haughton_filter
from faircanal.delivery import memory_steps


def test_block_length_matches_plant_step():
    assert config.DT_BLOCK_S == config.STEPS_PER_BLOCK * config.DT_PLANT_S


def test_settle_margin_covers_the_filter_memory():
    """The horizon margin must outlast the filter, for every candidate order.

    Derived from the filter rather than restated: a hard-coded number here
    would go stale the moment the filter order question is settled, and the
    failure it guards against - the filter tail falling off the end of the
    horizon - is silent.
    """
    for order in (HAUGHTON_FILTER["order_in_paper"],
                  HAUGHTON_FILTER["order_in_released_code"]):
        needed = memory_steps(haughton_filter(order))
        assert config.SETTLE_MARGIN_STEPS >= needed, (
            f"a filter of order {order} remembers {needed} steps, but the "
            f"horizon margin is only {config.SETTLE_MARGIN_STEPS}"
        )


def test_saturation_tolerance_above_solver_tolerance():
    """A user must never be declared saturated because of solver noise."""
    solver_tol = max(
        config.LP_OPTIONS["primal_feasibility_tolerance"],
        config.LP_OPTIONS["dual_feasibility_tolerance"],
    )
    assert config.EPS_SAT >= 10.0 * solver_tol


def test_ratio_is_capped():
    """Over-delivery must not be able to inflate the fairness measure."""
    assert config.R_CAP == 1.0


def test_solver_settings_are_serialisable():
    settings = config.solver_settings()
    assert settings["method"] == "highs"
    assert isinstance(settings["options"], dict)
    assert "threads" in settings
