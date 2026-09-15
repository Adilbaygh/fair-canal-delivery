#!/usr/bin/env python3
"""Relaxed alone, or relaxed together: which families have to give?

The certificate asks what the canal would have to give for every order to
be filled, and it asks it of all the families at once. The obvious design
is the other one - try each family on its own, report the cheapest - and
it was the first design here. It was discarded because it answers the
wrong question: a scenario can be infeasible under every single family
relaxed alone and feasible when two are relaxed together, so the
one-at-a-time search reports an impossibility that is not one.

That is a claim the article makes, so it is a claim this measures rather
than asserts. For a chosen scenario it runs the search both ways: each
relaxable family alone, then all of them at once.

    python scripts/check_one_at_a_time.py     # the scenario the claim is about

The default is not a point of the scan, and that is the finding rather
than an oversight. On the scan's own scenario the source is the only
thing short of what is needed, at every infeasible point, so relaxing C8
alone fills every order and the two searches agree:

    python scripts/check_one_at_a_time.py --only 0.3 --cap-scale 1 --demand-scale 1

The difference between the two searches appears where the canal rather
than the source is the tight thing, which is why the default squeezes the
conveyance to a tenth and raises the demand by fifteen per cent at 90 per
cent supply. That scenario is the one the adequacy audit found the original
mistake in, and it is pinned in ``tests/test_certificate.py`` as well, so
the article's sentence cannot quietly lose its evidence.

It exits 1 if the two searches agree at the scenario asked for - if some
family alone suffices, or if nothing is filled even jointly - because
then that scenario does not evidence the claim and the output says which
one does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faircanal.certificate import (  # noqa: E402
    CertificateError,
    elastic_relaxation,
    families,
)
from faircanal.config import (  # noqa: E402
    BLOCKS,
    DT_PLANT_S,
    LEAD_BLOCKS,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.control import control_law  # noqa: E402
from faircanal.delivery import FilterSpec  # noqa: E402
from faircanal.network import corning_cascade  # noqa: E402
from faircanal.plant import build_plant, horizon_for, response_map  # noqa: E402
from faircanal.programme import assemble  # noqa: E402
from faircanal.scenario import one_user_per_gate  # noqa: E402
from run_experiments import make_limits  # noqa: E402


def build(
    fraction: float,
    gate_head_m: float,
    cap_scale: float,
    demand_scale: float,
):
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
    scenario = one_user_per_gate(
        network,
        BLOCKS,
        make_limits(network, gate_head_m=gate_head_m, cap_scale=cap_scale),
        source_discharge_m3_s=fraction * network.aggregate_demand,
        window=(LEAD_BLOCKS * STEPS_PER_BLOCK, steps - 1),
        demand_scale=demand_scale,
        lead_blocks=LEAD_BLOCKS,
    )
    return assemble(scenario, mapping)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", type=float, default=0.90, metavar="F",
        help="source availability as a fraction of aggregate demand",
    )
    parser.add_argument(
        "--gate-head", type=float, default=0.10, metavar="M",
        help="head the check gates are read at; lower means narrower gates",
    )
    parser.add_argument(
        "--cap-scale", type=float, default=0.1, metavar="S",
        help="multiply every reach's conveyance and gate travel rate by S",
    )
    parser.add_argument(
        "--demand-scale", type=float, default=1.15, metavar="S",
        help="multiply every user's ordered volume by S",
    )
    args = parser.parse_args(argv)

    programme = build(
        args.only, args.gate_head, args.cap_scale, args.demand_scale
    )
    if (args.gate_head, args.cap_scale, args.demand_scale) != (0.10, 1.0, 1.0):
        print(
            f"a constructed scenario, not a scan point: gate head "
            f"{args.gate_head} m, conveyance x{args.cap_scale}, demand "
            f"x{args.demand_scale}\n"
        )
    groups = families(programme)
    print(f"supply {args.only:.0%} of aggregate demand, "
          f"{len(groups)} relaxable families\n")

    print("each family relaxed alone, every order to be filled")
    print(f"  {'family':<46} {'schedule?':>10}  {'it would have to give':>22}")
    alone_enough = []
    for group in groups:
        try:
            answer = elastic_relaxation(programme, allowed={group.name})
        except CertificateError as error:
            print(f"  {group.name:<46} {'refused':>10}  {error}")
            continue
        if answer.feasible:
            alone_enough.append(group.name)
        gave = answer.slacks.get(group.name, float("inf"))
        shown = "-" if not answer.feasible else f"{gave:.4g} {group.unit}"
        print(f"  {group.name:<46} {str(answer.feasible):>10}  {shown:>22}")

    joint = elastic_relaxation(programme)
    print("\nall of them relaxed together")
    print(f"  schedule? {joint.feasible}")
    for name in joint.binding_families:
        print(f"    {name:<46} +{joint.slacks[name]:.4g} {joint.units[name]}")

    print()
    if alone_enough:
        print(
            "At this supply level a single family is enough: "
            + ", ".join(alone_enough)
            + ".\nThis scenario does not evidence the article's sentence about "
              "families\nhaving to give together; the one the article is about is "
              "the default,\nwhich squeezes the conveyance rather than the source:\n"
              "    python scripts/check_one_at_a_time.py"
        )
        return 1
    if not joint.feasible:
        print(
            "Nothing is filled even with every family relaxed together, so this "
            "point\nsays nothing about one-at-a-time against jointly. Choose a "
            "less extreme\nsupply level."
        )
        return 1
    print(
        f"No single family admits a schedule with every order filled; "
        f"{len(joint.binding_families)} of the\n{len(groups)} have to give "
        f"together. This is the measurement behind the article's\nclaim that a "
        f"one-at-a-time search reports an impossibility that is not one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
