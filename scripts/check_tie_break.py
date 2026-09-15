#!/usr/bin/env python3
"""How much does the tie-break stage actually decide?

The lexicographic fraction vector is unique; the schedule that attains it
usually is not, and a linear-programming solver may return any vertex of
the optimal face. The study therefore adds a final stage that picks, among
all lexicographic optima, the one whose gates move least - and claims that
this is what makes the reported schedule reproducible.

That claim is a number, so this measures it. The same programme is solved
by two different simplex variants, with the tie-break and without, and the
two schedules are compared in the Euclidean norm. The fractions are
compared as well: those are unique at the optimum, so a difference there
would mean something worse than a tie.

    python scripts/check_tie_break.py
    python scripts/check_tie_break.py --only 0.5

Both distances are printed in the same format on purpose. An earlier
version printed the one without the tie-break to four decimal places and
the one with it in scientific notation, which made a distance of $10^{-5}$
look like a distance of zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faircanal.config import (  # noqa: E402
    BLOCKS,
    DT_PLANT_S,
    SETTLE_MARGIN_STEPS,
)
from faircanal.control import control_law  # noqa: E402
from faircanal.delivery import FilterSpec  # noqa: E402
from faircanal.leximin import refine, solve_leximin  # noqa: E402
from faircanal.network import corning_cascade  # noqa: E402
from faircanal.plant import build_plant, horizon_for, response_map  # noqa: E402
from faircanal.programme import assemble, movement_tie_break  # noqa: E402
from run_experiments import make_limits, scenario_for  # noqa: E402

METHODS = ("highs-ds", "highs-ipm")
SUPPLIES = (0.95, 0.75, 0.60, 0.55, 0.50)


def canal():
    """Build the plant and the response map once, for every supply level."""
    spec = FilterSpec(order=3, cutoff_rad_per_s=3.0e-3, sample_time_s=DT_PLANT_S)
    network = corning_cascade()
    plant = build_plant(network, spec)
    steps = horizon_for(BLOCKS, margin=SETTLE_MARGIN_STEPS)
    mapping = response_map(
        plant,
        BLOCKS,
        margin=SETTLE_MARGIN_STEPS,
        law=control_law(list(plant.design_models), plant.weights, horizon=steps),
    )
    return network, mapping


def gap(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", type=float, nargs="+", default=list(SUPPLIES), metavar="F",
        help="supply levels to measure, as fractions of aggregate demand",
    )
    args = parser.parse_args(argv)

    network, mapping = canal()
    limits = make_limits(network)

    print(f"solvers {' and '.join(METHODS)}, {len(args.only)} supply levels\n")
    print(f"  {'supply':>7}  {'stages':>6}  {'||z1-z2|| plain':>16}  "
          f"{'refined':>10}  {'fractions':>10}  {'cost':>10}  {'bound':>8}")
    for fraction in args.only:
        programme = assemble(scenario_for(fraction, network, limits), mapping)
        tie = movement_tie_break(programme)

        plain, refined = [], []
        for method in METHODS:
            staged = solve_leximin(programme.ratio, method=method)
            plain.append(staged)
            refined.append(refine(programme.ratio, staged, tie, method=method))

        print(
            f"  {fraction:>6.0%}  {len(plain[0].stage_levels):>6}  "
            f"{gap(plain[0].z, plain[1].z):>16.3e}  "
            f"{gap(refined[0].z, refined[1].z):>10.3e}  "
            f"{gap(refined[0].ratios, refined[1].ratios):>10.3e}  "
            f"{refined[0].tie_break_cost:>10.4f}  "
            f"{refined[0].accuracy_bound:>8.1e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
