"""How wide does the storage-to-level reconciliation band C9' have to be?

C9' ties the slow (volume) description of a pool's storage to the fast
(level) description:

    | S_p[j] - S_p^0 - A_p * ybar_p[j] |  <=  eps_p

Two things in that row have no published value: the area A_p and the width
eps_p. This script measures both rather than assuming them, because the
answer turned out to be an order of magnitude away from the first guess.

WHICH AREA
    The model document writes A_p as the water-surface area, top width
    times length. But the identified pool does not integrate its net flow
    at that rate: the integrator gain of the third-order model is
    dt / A_d with A_d the backwater area, and for this canal the two
    differ by between 13 and 76 per cent. Both are reported below.

WHY THE TWO ACCOUNTS CANNOT AGREE EXACTLY
    An IDZ pool is not a level pool. Its storage is not the downstream
    level times an area, because the surface tilts when the flow changes
    and the wedge that tilting stores is exactly what the model's delay
    and zero represent. So the width of the band is a measurement of the
    model, not a tolerance anybody is free to choose, and reporting it is
    more useful than hiding it.

Run from the project root:

    python scripts\\check_storage_band.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))

from faircanal.closedloop import simulate_closed_loop  # noqa: E402
from faircanal.config import (  # noqa: E402
    BAND_TOLERANCE_M,
    BLOCKS,
    DT_PLANT_S,
    LEAD_BLOCKS,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.control import control_law  # noqa: E402
from faircanal.delivery import FilterSpec  # noqa: E402
from faircanal.leximin import solve_leximin  # noqa: E402
from faircanal.network import corning_cascade, surface_area  # noqa: E402
from faircanal.plant import build_plant, horizon_for, response_map  # noqa: E402
from faircanal.programme import assemble  # noqa: E402
from faircanal.scenario import one_user_per_gate  # noqa: E402
from run_experiments import make_limits  # noqa: E402

FILTER_ORDER = 3
CUTOFF_RAD_PER_S = 3.0e-3
SCAN = (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50)

#: A band wide enough that C9' cannot bind, so that what is measured is
#: the model's discrepancy and not the constraint's own width. Ten metres
#: of water column is four times the deepest reach.
WIDE_BAND_M = 10.0


def backwater_areas(models) -> np.ndarray:
    """The area the identified pool actually integrates its net flow at.

    The third-order model's integrator gain is (b1 - b2 + b3) / (1 - a2)
    metres per cubic metre per second per step, and that gain is dt / A_d
    by construction, so A_d comes straight back out of the coefficients.
    """
    return np.array(
        [
            DT_PLANT_S * (1.0 - pool.alpha[1]) / (pool.b[0] - pool.b[1] + pool.b[2])
            for pool in models
        ]
    )


def accounts(run, taus, area, steps, blocks, per_block):
    """The two storage accounts, per pool and per block, in cubic metres.

    Both are deviations from the nominal operating point, so both are zero
    for a run at rest and no absolute storage is needed.

    ``applied[k, i]`` is the flow through gate ``i+1``, which is the inflow
    to pool ``i+1``; that pool's outflow is the flow through gate ``i``,
    and the most downstream pool has none. It is the source's convention
    and the one ``closedloop`` simulates.
    """
    size = area.size
    volume = np.zeros((size, blocks + 1))
    level = np.zeros((size, blocks + 1))
    for i in range(size):
        tau = taus[i]
        inflow = np.zeros(steps)
        if tau < steps:
            inflow[tau:] = run.applied[: steps - tau, i]
        outflow = run.applied[:, i - 1] if i >= 1 else np.zeros(steps)
        net = (inflow - outflow - run.offtake_applied[:, i]) * DT_PLANT_S
        for j in range(blocks):
            first = j * per_block
            volume[i, j + 1] = volume[i, j] + net[first : first + per_block].sum()
        for j in range(blocks + 1):
            first = min(j * per_block, steps - per_block)
            level[i, j] = area[i] * run.levels[first : first + per_block, i].mean()
    return volume, level


def main() -> int:
    spec = FilterSpec(
        order=FILTER_ORDER,
        cutoff_rad_per_s=CUTOFF_RAD_PER_S,
        sample_time_s=DT_PLANT_S,
    )
    network = corning_cascade()
    plant = build_plant(network, spec)
    size = network.size
    steps = horizon_for(BLOCKS, margin=SETTLE_MARGIN_STEPS)
    per_block = STEPS_PER_BLOCK
    law = control_law(list(plant.design_models), plant.weights, horizon=steps)
    mapping = response_map(plant, BLOCKS, margin=SETTLE_MARGIN_STEPS, law=law)

    taus = [pool.tau for pool in plant.plant_models]
    surface = np.array([surface_area(reach) for reach in network.reaches])
    backwater = backwater_areas(plant.plant_models)

    print(f"canal: {network.name}, {size} pools, {steps} steps, {BLOCKS} blocks")
    print()
    print(f"{'pool':>4} {'A_s surface':>12} {'A_d backwater':>14} {'ratio':>7}")
    for i in range(size):
        print(
            f"{i + 1:>4} {surface[i]:>12.0f} {backwater[i]:>14.0f} "
            f"{surface[i] / backwater[i]:>7.3f}"
        )
    print()

    # The canal the scan solves, gates and all - the same function, not a
    # second copy of it. A measurement taken on a slightly different canal
    # than the one the study reports is not a measurement of anything the
    # study reports.
    #
    # With one deliberate difference: the band itself is opened so wide
    # that it cannot bind. C9' is in the programme now, so measuring the
    # discrepancy on a schedule the programme has already forced to keep
    # that discrepancy under 0.15 m would return "under 0.15 m" whatever
    # the truth was, and would do it convincingly. The criterion is given
    # the freedom to violate the band, and then asked how far it went.
    limits = make_limits(network, band_tolerance_m=WIDE_BAND_M)
    window = (LEAD_BLOCKS * per_block, steps - 1)

    rest = np.zeros((steps, size))
    base = simulate_closed_loop(plant.loop, np.zeros(size), rest, law=law)
    zero_surface = accounts(base, taus, surface, steps, BLOCKS, per_block)
    zero_backwater = accounts(base, taus, backwater, steps, BLOCKS, per_block)
    residual = max(
        np.abs(zero_surface[0] - zero_surface[1]).max(),
        np.abs(zero_backwater[0] - zero_backwater[1]).max(),
    )
    print(f"discrepancy at rest (must be zero): {residual:.3e} m^3")
    print()

    hold = np.repeat(np.eye(BLOCKS), per_block, axis=0)
    print("discrepancy on the schedule the criterion actually chooses")
    print(f"{'Q %':>5} {'min r':>9} {'A_s [m]':>10} {'A_d [m]':>10}")
    peaks = {"surface": 0.0, "backwater": 0.0}
    for fraction in SCAN:
        scenario = one_user_per_gate(
            network,
            BLOCKS,
            limits,
            source_discharge_m3_s=fraction * network.aggregate_demand,
            window=window,
            lead_blocks=LEAD_BLOCKS,
            settle_margin=SETTLE_MARGIN_STEPS,
        )
        programme = assemble(scenario, mapping)
        try:
            answer = solve_leximin(programme.ratio)
        except Exception as error:  # a point with no schedule says nothing here
            print(f"{fraction * 100:>5.0f} {'none':>9}   ({type(error).__name__})")
            continue
        orders = np.asarray(answer.z).reshape(size, BLOCKS)
        drive = np.zeros((steps, size))
        for i in range(size):
            drive[: BLOCKS * per_block, i] = hold @ orders[i]
        run = simulate_closed_loop(plant.loop, np.zeros(size), drive, law=law)

        errors = {}
        for label, area, zero in (
            ("surface", surface, zero_surface),
            ("backwater", backwater, zero_backwater),
        ):
            volume, level = accounts(run, taus, area, steps, BLOCKS, per_block)
            gap = (volume - zero[0]) - (level - zero[1])
            errors[label] = np.abs(gap / area[:, None]).max()
            peaks[label] = max(peaks[label], errors[label])
        print(
            f"{fraction * 100:>5.0f} {answer.ratios.min():>9.5f} "
            f"{errors['surface']:>10.5f} {errors['backwater']:>10.5f}"
        )

    print()
    print(f"worst over the scan, surface area   : {peaks['surface']:.4f} m")
    print(f"worst over the scan, backwater area : {peaks['backwater']:.4f} m")
    print(f"the band the study uses             : {BAND_TOLERANCE_M:.4f} m")
    print()
    print(
        "The backwater area is the right coefficient: it is what the pool "
        "integrates at,\nand the width it needs does not drift with scarcity "
        "the way the surface area's does."
    )
    if peaks["backwater"] > BAND_TOLERANCE_M:
        print()
        print(
            f"The band is too narrow: the two accounts differ by "
            f"{peaks['backwater']:.4f} m somewhere on this scan and the "
            f"programme allows {BAND_TOLERANCE_M:.4f} m. Raise "
            f"BAND_TOLERANCE_M to at least the measured figure, or C9' is "
            f"constraining the schedule rather than describing the model."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
