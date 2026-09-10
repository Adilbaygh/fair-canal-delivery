"""Permanent tests for the frozen settings. Never deleted.

A silent change to a tolerance or a discretisation constant changes every
published number without changing a single line of the model, which is why
these are asserted rather than trusted.
"""

from __future__ import annotations

from faircanal import config


def test_block_length_matches_plant_step():
    assert config.DT_BLOCK_S == config.STEPS_PER_BLOCK * config.DT_PLANT_S


def test_settle_margin_covers_filter_memory():
    """The filter impulse response stays above 1e-4 for 81 minutes."""
    assert config.SETTLE_MARGIN_STEPS >= 81


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
