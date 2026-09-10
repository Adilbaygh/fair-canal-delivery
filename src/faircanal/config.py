"""Frozen numerical settings.

Every number this package publishes is produced with the constants in this
module. They are not defaults to be overridden at the call site: changing one
changes the published result, so a change here is a change to the study.

The values are recorded verbatim in results/environment.json on every run,
so the article and the archive always report the same configuration in the
same words.
"""

from __future__ import annotations

from types import MappingProxyType

# --------------------------------------------------------------------------
# Time discretisation
# --------------------------------------------------------------------------

#: Plant and filter step [s]. The source pool models are identified at a
#: one-minute sample time; this is not a free parameter.
DT_PLANT_S: float = 60.0

#: Decision block length [s]. Orders are reshaped block by block and held
#: zero-order over the plant steps inside a block.
DT_BLOCK_S: float = 900.0

#: Plant steps per decision block.
STEPS_PER_BLOCK: int = 15

#: Horizon margin after the last delivery window [plant steps].
#:
#: The filter tail has to fit inside the horizon, or part of every order
#: falls off the end and the volume balance stops adding up without
#: anything complaining. The margin therefore has to cover the memory of
#: whichever filter order is in use, and the two candidate orders differ:
#: the third-order filter needs 82 steps and the fourth-order one needs
#: 103. The margin covers both with room to spare, so the horizon does not
#: have to be revisited when the order question is settled. A longer
#: horizon costs solver time and nothing else.
#:
#: test_settle_margin_covers_the_filter_memory derives the requirement from
#: the filter itself rather than restating these numbers.
SETTLE_MARGIN_STEPS: int = 120

assert abs(DT_BLOCK_S - STEPS_PER_BLOCK * DT_PLANT_S) < 1e-12, (
    "DT_BLOCK_S must equal STEPS_PER_BLOCK * DT_PLANT_S"
)

# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

#: scipy.optimize.linprog method. HiGHS is used through scipy so that the
#: constraint matrices stay ours and no canonicalisation layer sits between
#: the written model and the solved one.
LP_METHOD: str = "highs"

#: Frozen solver options. Everything not named here is the solver default,
#: and no thread count is set.
LP_OPTIONS = MappingProxyType(
    {
        "presolve": True,
        "primal_feasibility_tolerance": 1.0e-9,
        "dual_feasibility_tolerance": 1.0e-9,
        "disp": False,
    }
)

#: Saturation tolerance of the lexicographic stage test. Chosen three orders
#: of magnitude above the solver feasibility tolerance so that a user is
#: never declared saturated because of solver noise. The price is that the
#: returned solution can differ from the lexicographic optimum by at most
#: n_users * EPS_SAT; that bound is reported, and the word "optimal" is not
#: used without it.
EPS_SAT: float = 1.0e-6

#: Tolerance used when asserting that a computed quantity satisfies a
#: constraint. An inequality violation reported with this label is an
#: inequality violation, never an equality residual.
EPS_FEAS: float = 1.0e-9

# --------------------------------------------------------------------------
# Model constants
# --------------------------------------------------------------------------

#: Delivery allowance on the released-volume budget. With no allowance a
#: ratio of exactly one is unreachable in principle, because the filter tail
#: leaves the window and a non-negative order cannot cancel it.
GAMMA_DELIVERY: float = 0.05

#: The delivered-volume ratio is capped at one, so over-delivery cannot
#: inflate the fairness measure.
R_CAP: float = 1.0

#: Users whose ordered volume is below this bound [m^3] are excluded, because
#: the ratio has the ordered volume in its denominator. Exclusions are
#: reported as a decision record, one line per excluded user with the reason.
D_MIN_M3: float = 1.0

#: Gravitational acceleration [m/s^2].
G_ACCEL: float = 9.81

#: Gate discharge coefficient.
C_D: float = 0.61


def solver_settings() -> dict:
    """Return the frozen solver configuration as a plain, sortable dict."""
    return {
        "method": LP_METHOD,
        "options": dict(LP_OPTIONS),
        "eps_sat": EPS_SAT,
        "eps_feas": EPS_FEAS,
        "threads": "not set (solver default)",
    }


def discretisation_settings() -> dict:
    """Return the frozen time discretisation as a plain, sortable dict."""
    return {
        "dt_plant_s": DT_PLANT_S,
        "dt_block_s": DT_BLOCK_S,
        "steps_per_block": STEPS_PER_BLOCK,
        "settle_margin_steps": SETTLE_MARGIN_STEPS,
    }
