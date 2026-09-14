"""Frozen numerical settings.

Every number this package publishes is produced with the constants in this
module. They are not defaults to be overridden at the call site: changing one
changes the published result, so a change here is a change to the study.

The values are recorded verbatim in results/environment.json on every run,
so the article and the archive always report the same configuration in the
same words.

A constant that a published run does not actually use does not belong here.
That rule was broken once and it cost an audit finding: GAMMA_DELIVERY read
0.05 while every scan ran at zero, so a reader of this module would have
reported a number the results contradict. The constants below are therefore
the ones the driver imports, not a wish list, and the scenario block at the
end holds the values that used to sit in scripts/run_experiments.py.
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
#: (n_users + 1) * EPS_SAT -- one for each stage that may freeze a user early,
#: and one more for the tie-break stage, which lets every user fall by that
#: much below the level it reached. That bound is reported, and the word
#: "optimal" is not used without it.
EPS_SAT: float = 1.0e-6

#: Tolerance used when asserting that a computed quantity satisfies a
#: constraint. An inequality violation reported with this label is an
#: inequality violation, never an equality residual.
EPS_FEAS: float = 1.0e-9

# --------------------------------------------------------------------------
# Model constants
# --------------------------------------------------------------------------

#: Delivery allowance on the released-volume budget, the gamma_i of C3.
#:
#: Zero is the strict reading: what is released inside the window is what
#: counts, with nothing forgiven. It is also the least flattering choice,
#: because the filter's tail leaves the window and a non-negative order
#: cannot cancel it, so a fraction of exactly one is hard to reach rather
#: than free. The main scan runs at zero; the budget sensitivity runs at
#: 0.5 and is reported separately.
GAMMA_DELIVERY: float = 0.0

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

#: Conveyance efficiency of a reach, the eta_p of C9.
#:
#: Held at one, which is the open declaration of the loss question rather
#: than an answer to it: the Kostiakov-type form the literature gives is
#: concave, so the useful flow is convex and a row bounding an outflow by it
#: would describe a non-convex set. Until a formula can be read from its
#: source, the losses stay outside the programme and the value is published
#: as an assumption, not buried as a default.
CONVEYANCE_EFFICIENCY: float = 1.0

# --------------------------------------------------------------------------
# Scenario constants
# --------------------------------------------------------------------------
#
# These decide what the instance is, not how it is solved. They used to live
# in scripts/run_experiments.py, where the module contract above could not
# reach them and where the provenance table of the model document did not
# carry them either. Each one is published with its provenance.

#: Ordering blocks in a run. *assumed*
BLOCKS: int = 8

#: Blocks of notice before the source restriction begins. *assumed*
LEAD_BLOCKS: int = 2

#: Plant steps over which the source profile ramps. *assumed*
SOURCE_RAMP_STEPS: int = 15

#: Plant steps over which C7, the level band, is relaxed at the start of a
#: run. *assumed*
#:
#: The exemption covers C7 and nothing else. An earlier version of the
#: assembly shared one step index between C5, C6 and C7, which quietly freed
#: the conveyance and the gate travel rate over the same window; the level
#: band is the one that the order-induced transient can legitimately break,
#: and it is the one the model document names.
WARM_UP_STEPS: int = 15

#: Gate travel rate as a fraction of the reach's conveyance, per step.
#: *assumed*
TRAVEL_FRACTION: float = 0.25

#: A user's own outlet flow limit, as a multiple of its nominal draw, the
#: v_bar_i of C1. *assumed*
#:
#: No offtake geometry is published for this canal: Table 6 of the source
#: gives the offtake discharges and Table 5 the check gates, so the outlet
#: itself has to be sized by assumption. A headroom of one half is the
#: ordinary allowance in irrigation practice, and it is the half of C1 that
#: makes the structural ceiling of Section 2.8 capable of falling below one:
#: with no outlet limit at all, every ceiling is one by construction and the
#: diagnostic cannot report the case it exists for. Reported with a
#: sensitivity over 1.25, 1.5 and 2.0.
OUTLET_HEADROOM: float = 1.5

#: Head difference across a check gate at its saturation point [m].
#: *assumed*
#:
#: The orifice relation C_d W H sqrt(2 g dH) needs a head drop, and the
#: design levels it would come from are not published. Ten centimetres is
#: the ordinary design value for a check structure on a canal of this size.
#: It matters: at this head the gates of the upper reaches pass less than
#: the reaches convey, so the gate is what binds rather than the channel.
#: Reported with a sensitivity over 0.05, 0.10 and 0.20.
GATE_HEAD_M: float = 0.10

#: First block a user may re-shape, the j_ann_i of C4. *derived*
#:
#: Before this block the order stays at its announced nominal value -- which
#: is causality, not a preference, and is why C4 pins the deviation rather
#: than the order itself.
#:
#: It cannot be chosen freely, and the first value tried was wrong. The
#: restriction takes effect at LEAD_BLOCKS, so a user whose orders are
#: frozen until then is told about the shortage at the moment it arrives
#: and has no notice at all; with the filter needing some eighty steps
#: between an order and the water, no schedule exists and the programme is
#: infeasible from 95 per cent downwards. Measured: at two blocks only full
#: supply survives, at one block and at zero the feasible range is the same
#: eleven points, 100 down to 50 per cent. One block is therefore the most
#: notice the instance can be asked to give while leaving C4 doing real
#: work, and the rule behind it -- the announcement must precede the
#: restriction -- is enforced in scenario.one_user_per_gate rather than
#: left to be rediscovered.
ANNOUNCE_BLOCKS: int = 1

#: Width of the storage-to-level reconciliation band C9', expressed as the
#: level error it tolerates [m]. *derived, by measurement*
#:
#: The band ties the slow volume account to the fast level account:
#: epsilon_p = A_d_p * BAND_TOLERANCE_M. An equality between them would be
#: defeated at once, because an IDZ pool is not a level pool: its storage is
#: not the downstream level times an area, since the surface tilts when the
#: flow changes and the wedge that tilting holds is exactly what the model's
#: delay and zero represent.
#:
#: This value is therefore not chosen. scripts/check_storage_band.py runs
#: the closed loop on the schedule the criterion itself selects at each
#: point of the scan and reports how far the two accounts drift apart: zero
#: at rest, and between 0.069 and 0.121 m of level across the scan. Fifteen
#: centimetres covers that with room, and the measurement is rerun whenever
#: the programme changes, because a band narrower than the model's own
#: inconsistency would make the programme infeasible for a reason that has
#: nothing to do with water. Reported with a sensitivity over 0.10, 0.15
#: and 0.25.
BAND_TOLERANCE_M: float = 0.15

#: Area used in C9' to convert a level into a storage [m^2 per m].
#:
#: "backwater" is the area the identified pool integrates its net flow at,
#: recovered from the model's own coefficients as dt (1 - alpha_2) /
#: (b_1 - b_2 + b_3). "surface" is the water-surface area, top width times
#: length, which is what the model document wrote and which is not the same
#: quantity: on this canal the two differ by between 13 and 76 per cent, and
#: the surface area's reconciliation error grows with scarcity while the
#: backwater area's does not.
BAND_AREA: str = "backwater"


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


def scenario_settings() -> dict:
    """Return the frozen scenario choices as a plain, sortable dict.

    Everything here is *assumed* except the two model constants at the end,
    and every entry reaches both results/environment.json and the digest
    carried inside each certificate, so a result can always be read back
    against the instance that produced it.
    """
    return {
        "blocks": BLOCKS,
        "lead_blocks": LEAD_BLOCKS,
        "source_ramp_steps": SOURCE_RAMP_STEPS,
        "warm_up_steps": WARM_UP_STEPS,
        "travel_fraction": TRAVEL_FRACTION,
        "outlet_headroom": OUTLET_HEADROOM,
        "gate_head_m": GATE_HEAD_M,
        "announce_blocks": ANNOUNCE_BLOCKS,
        "band_tolerance_m": BAND_TOLERANCE_M,
        "band_area": BAND_AREA,
        "gamma_delivery": GAMMA_DELIVERY,
        "conveyance_efficiency": CONVEYANCE_EFFICIENCY,
        "ratio_cap": R_CAP,
        "demand_floor_m3": D_MIN_M3,
    }
