#!/usr/bin/env python3
"""Which programme's prices does the certificate report?

The certificate names the families that stand in the way and says what a
unit of each is worth. A price only means something with respect to an
objective, and the staged procedure solves several: the first stage sets
the worst-off user's level, each later stage raises somebody already above
it, and the tie-break minimises gate movement over the fair optima. Three
different objectives, three different sets of duals, one field called
``marginals``.

This prints all three side by side for one supply level, so the choice can
be made on the evidence rather than on which stage happened to be last.

    python scripts/check_duals.py --only 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faircanal.certificate import families  # noqa: E402
from faircanal.config import (  # noqa: E402
    BLOCKS,
    DT_PLANT_S,
    SETTLE_MARGIN_STEPS,
)
from faircanal.control import control_law  # noqa: E402
from faircanal.delivery import FilterSpec  # noqa: E402
from faircanal.leximin import (  # noqa: E402
    _solve,
    _stage_programme,
    refine,
    solve_leximin,
)
from faircanal.network import corning_cascade  # noqa: E402
from faircanal.plant import build_plant, horizon_for, response_map  # noqa: E402
from faircanal.programme import assemble, movement_tie_break  # noqa: E402
from run_experiments import make_limits, scenario_for  # noqa: E402


def build(fraction: float):
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
    scenario = scenario_for(fraction, network, make_limits(network))
    return assemble(scenario, mapping)


def by_family(programme, marginals: np.ndarray) -> dict[str, tuple[int, float]]:
    rows = programme.ratio.a_ub.shape[0]
    marginals = np.asarray(marginals, dtype=float)[:rows]
    out = {}
    for group in families(programme):
        prices = np.abs(marginals[group.first : group.last])
        active = int((prices > 1e-9).sum())
        if active:
            out[group.name] = (active, float(prices.max()))
    return out


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", type=float, default=0.5, metavar="F")
    args = parser.parse_args(argv)

    programme = build(args.only)
    ratio = programme.ratio

    first = _solve(
        *_stage_programme(ratio, list(range(ratio.n_users)), {}),
        what="first stage",
    )
    staged = solve_leximin(ratio)
    refined = refine(ratio, staged, movement_tie_break(programme))

    views = (
        ("first stage (sets the worst-off level)", first.ineqlin.marginals),
        (f"last stage ({len(staged.stage_levels)} of them)", staged.marginals),
        ("tie-break (least gate movement)", refined.marginals),
    )

    print(f"supply {args.only:.0%} of aggregate demand\n")
    for title, marginals in views:
        print(f"  {title}")
        found = by_family(programme, marginals)
        if not found:
            print("    nothing priced above 1e-9")
        for name, (active, price) in found.items():
            print(f"    {name:48} {active:3} rows   {price:.4g}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
