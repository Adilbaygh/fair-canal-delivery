#!/usr/bin/env python3
"""Run the scan the pre-registration fixed, and write down what it fixed.

Every parameter here comes from the study's pre-registration, which was
written and frozen before any of this was run, and which states each value
with its provenance: measured, derived, or assumed. Nothing in this file
chooses anything: it builds the canal, walks the scarcity scan, solves the
alternatives on the same feasible set at each point, and writes the
numbers that document said would be reported - including the ones that
would embarrass the method if they came out the wrong way.

What it writes
--------------
``results/scan/<label>/Q<pct>.json``
    One file per scan point, with everything: the fractions of every
    variant, the lexicographic stage levels and who saturated at each,
    the structural ceilings, the binding families with their shadow
    prices, the elastic relaxation in its own units, the certificate and
    its digest, and how long each solve took.

``results/tables/<label>_summary.csv``
    One row per point and variant: worst fraction, total, spread.

``results/tables/<label>_ratios.csv``
    One row per point, variant and user: the fraction that user received.

``results/tables/<label>_certificates.csv``
    One row per point: what the certificate says, in fields.

The tables are rebuilt from the per-point files every run, so they are
complete whether the scan ran in one sitting or five.

Running it in the time a person will actually wait
--------------------------------------------------
The free-gate bound M1 is a hundred times the size of everything else and
takes minutes a point. So:

* every point is written the moment it finishes, and a later run picks up
  where an earlier one stopped - per *variant*, not per point, so adding
  M1 later does not re-solve the rest;
* ``--skip M1`` leaves it out, which turns the whole scan from hours into
  minutes and is the right first pass;
* progress is printed as it happens, with the time each solve took.

Interrupting it with Ctrl+C loses only the point in flight. It can take a
few seconds to stop, because it waits for the linear programme in front of
it to finish.

The sensitivity runs
--------------------
H5, H7 and H9 are the same scan with one thing changed, so they are this
script with a flag and their own label rather than a second script that
could drift from it:

    --label order4  --filter-order 4        H7, the filter order
    --label cut1e3  --cutoff 1e-3           H9, the cut-off
    --label budget  --overshoot 0.5         H5, the volume budget

H8, the wave-damping estimate, needs a hook in the identification that
does not exist yet, and is not pretended here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.baselines import (  # noqa: E402
    BaselineError,
    leximin,
    min_spread,
    time_shift,
    unchanged_order,
    upper_bound,
    upper_bound_level,
    utilitarian,
)
from faircanal.certificate import certify  # noqa: E402
from faircanal.config import (  # noqa: E402
    DT_PLANT_S,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.control import control_law  # noqa: E402
from faircanal.delivery import FilterSpec, memory_steps  # noqa: E402
from faircanal.geometry import uniform_discharge  # noqa: E402
from faircanal.config import LP_METHOD, LP_OPTIONS  # noqa: E402
from faircanal.leximin import (  # noqa: E402
    LeximinError,
    SolverUndecided,
    solve_leximin,
)
from faircanal.network import corning_cascade  # noqa: E402
from faircanal.plant import build_plant, horizon_for, response_map  # noqa: E402
from faircanal.programme import assemble  # noqa: E402
from faircanal.provenance import repo_root, write_json, write_text  # noqa: E402
from faircanal.scenario import Limits, one_user_per_gate  # noqa: E402

# --- what the pre-registration fixed --------------------------------------
# Sections 1.1 to 1.5 of it. These are not defaults to be tuned; changing
# one means the run is no longer the pre-registered one, which is why every
# override says so on the way past.
BLOCKS = 8
LEAD_BLOCKS = 2
WARM_UP_STEPS = 15
TRAVEL_FRACTION = 0.25
FILTER_ORDER = 3
CUTOFF_RAD_PER_S = 3.0e-3
OVERSHOOT = 0.0
SCAN = tuple(round(1.00 - 0.05 * step, 2) for step in range(15))  # 100% .. 30%

CODES = ("B1", "B2", "B3", "B4", "B5", "M1")


# ---------------------------------------------------------------------------
# Building the canal, once
# ---------------------------------------------------------------------------


def make_limits(network) -> Limits:
    """Section 1.4: conveyance derived, the rest declared as assumed."""
    full = [
        uniform_discharge(reach.pool, reach.pool.canal_depth_m)
        for reach in network.reaches
    ]
    return Limits(
        capacity_m3_s=tuple(full),
        nominal_m3_s=network.steady_discharges,
        level_band_m=tuple(
            (
                -(reach.pool.canal_depth_m - reach.pool.target_level_m),
                reach.pool.canal_depth_m - reach.pool.target_level_m,
            )
            for reach in network.reaches
        ),
        travel_rate_m3_s=tuple(TRAVEL_FRACTION * value for value in full),
        warm_up_steps=WARM_UP_STEPS,
    )


def build(order: int, cutoff: float, margin: int | None = None):
    """The plant and its response map: the expensive part, done once.

    ``margin`` is the settling tail after the last order block, and it is
    not a free parameter: the filter's memory sets a floor under it, and
    a run whose margin is below that floor loses part of the last block's
    water off the end of the horizon. The frozen value covers the filter
    the study was pre-registered with.

    It has to be adjustable all the same, because the sensitivity runs
    change the filter. At a cut-off of one thousandth of a radian a
    second the memory is 193 steps against a frozen margin of 120, so
    that run cannot be performed at all without a longer tail - and the
    honest way to lengthen it is to say so on the command line and have
    it recorded with the results, rather than to leave the hypothesis
    untested or, worse, to weaken the check that caught it.
    """
    spec = FilterSpec(
        order=order, cutoff_rad_per_s=cutoff, sample_time_s=DT_PLANT_S
    )
    # Before anything expensive: the filter alone decides whether this run
    # is possible, and it costs nothing to ask. Building the canal first
    # would spend ten seconds to arrive at the same refusal.
    needed = memory_steps(spec)
    if margin is None:
        margin = SETTLE_MARGIN_STEPS
    if margin < needed:
        raise SystemExit(
            f"this filter remembers for {needed} steps and the settling margin "
            f"is {margin}. Pass --margin {needed} or more; the run would "
            f"otherwise lose the last block's water off the end of the horizon."
        )
    network = corning_cascade()
    plant = build_plant(network, spec)
    steps = horizon_for(BLOCKS, margin=margin)
    mapping = response_map(
        plant,
        BLOCKS,
        margin=margin,
        law=control_law(list(plant.design_models), plant.weights, horizon=steps),
    )
    return network, plant, mapping


# ---------------------------------------------------------------------------
# Turning answers into records
# ---------------------------------------------------------------------------


def plain(value):
    """JSON that a reader and a diff can both make sense of."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def describe(answer, names, seconds: float) -> dict:
    record = {
        "status": "solved",
        "feasible": True,
        "name": answer.name,
        "ratios": {name: float(value) for name, value in zip(names, answer.ratios)},
        "worst": answer.worst,
        "total": answer.total,
        "spread": answer.spread,
        "fulfilled": answer.fulfilled,
        "n_programmes": answer.n_programmes,
        "seconds": round(seconds, 3),
        "detail": plain(answer.detail),
    }
    return plain(record)


def describe_certificate(certificate) -> dict:
    return plain(
        {
            "fulfilled": certificate.fulfilled,
            "ratios": certificate.ratios,
            "shortfalls_m3": certificate.shortfalls,
            "ceilings": certificate.ceilings,
            "impossible_users": certificate.impossible_users,
            "stage_levels": certificate.stage_levels,
            "saturated": certificate.saturated,
            "binding": [
                {
                    "family": entry.family,
                    "active_rows": entry.active_rows,
                    "largest_price": entry.largest_price,
                    "unit": entry.unit,
                }
                for entry in certificate.binding
            ],
            "relaxation": {
                "feasible": certificate.relaxation.feasible,
                "cost": certificate.relaxation.cost,
                "slacks": certificate.relaxation.slacks,
                "units": certificate.relaxation.units,
                "binding_families": certificate.relaxation.binding_families,
            },
            "digest": certificate.digest,
            "report": certificate.report(),
        }
    )


# ---------------------------------------------------------------------------
# One scan point
# ---------------------------------------------------------------------------


def run_point(
    fraction: float,
    network,
    plant,
    mapping,
    limits,
    wanted: tuple[str, ...],
    record: dict,
    overshoot: float,
    settle_margin: int = SETTLE_MARGIN_STEPS,
    bound_method: str | None = None,
    bound_options: dict | None = None,
    bound_level_only: bool = False,
) -> dict:
    """Fill in whatever this point is still missing, and say what it did."""
    # The delivery window is the one agreed with the farmers, and it does
    # not move when the filter gets slower. That is the whole premise: the
    # question this study exists to answer is whether the water arrives
    # inside the agreed window, so a window that stretched to accommodate
    # a slower filter would answer it by definition.
    #
    # It has to be said explicitly because the scenario's default is the
    # end of the horizon, and the horizon has to grow to hold a longer
    # filter's tail. Left to the default, a longer tail lengthened the
    # window, the window lengthened the demand - nominal draw across the
    # window is what demand means - and the same eight order blocks were
    # asked to fill 35% more water. Feasibility collapsed, and it would
    # have been read as the filter's doing. Measured: at a margin of 193
    # the demand rose from 138,600 to 186,780 cubic metres with no change
    # to the filter at all.
    window = (
        LEAD_BLOCKS * STEPS_PER_BLOCK,
        horizon_for(BLOCKS, STEPS_PER_BLOCK, SETTLE_MARGIN_STEPS) - 1,
    )
    scenario = one_user_per_gate(
        network,
        BLOCKS,
        limits,
        source_discharge_m3_s=fraction * network.aggregate_demand,
        window=window,
        lead_blocks=LEAD_BLOCKS,
        overshoot=overshoot,
        settle_margin=settle_margin,
    )
    programme = assemble(scenario, mapping)
    names = programme.ratio.names
    variants = record.setdefault("variants", {})

    record["fraction"] = fraction
    record["source_m3_s"] = float(scenario.source_discharge_m3_s)
    record["users"] = list(names)
    record["blocks"] = BLOCKS
    record["horizon_steps"] = programme.steps

    # A variant that stopped on a limit is not done: it has no answer, so a
    # later run with a longer budget or a different solver has to try again.
    missing = [
        code
        for code in wanted
        if code not in variants or variants[code].get("status") == "undecided"
    ]
    if not missing and "certificate" in record:
        print(f"  Q={fraction:>5.0%}  already done", flush=True)
        return record

    first = None
    empty = False      # this polytope is provably empty
    undecided = False  # the projection came back without a verdict
    if {"B1", "B2", "B5"} & set(missing):
        clock = time.perf_counter()
        try:
            first = unchanged_order(programme)
        except SolverUndecided as error:
            # The solver returned without deciding. That is a statement
            # about one solve and not about the feasible set, so nothing
            # here may be written down as infeasible.
            #
            # Two variants lose their answer and only two: B1, which is
            # this solve, and B2, which is defined as B1's answer shifted
            # in time and so has nothing to shift. B3, B4 and B5 are
            # separate programmes on the same polytope - B5 takes the
            # projection only as a starting point and does without it -
            # and they are attempted, because refusing to attempt them
            # would record five verdicts on the strength of one solve
            # that produced none.
            elapsed = time.perf_counter() - clock
            undecided = True
            for code in ("B1", "B2"):
                if code not in missing:
                    continue
                variants[code] = {
                    "status": "undecided",
                    "feasible": None,
                    "solver_status": getattr(error, "status", -1),
                    "why": (
                        str(error)
                        if code == "B1"
                        else "B2 is B1's answer shifted one block early, and "
                        f"B1 has no answer: {error}"
                    ),
                    "seconds": round(elapsed, 3) if code == "B1" else 0.0,
                }
            print(
                f"  Q={fraction:>5.0%}  B1 NOT SOLVED - the solver returned "
                f"without deciding after {elapsed:.0f}s. This is not "
                f"infeasibility.",
                flush=True,
            )
        except BaselineError as error:
            # Nothing on *this* polytope can be solved. Every variant that
            # lives on it is infeasible, and the certificate is the whole
            # answer for them.
            #
            # M1 is the exception and the distinction is not pedantic: it
            # frees the gate commands, so it is a different feasible set,
            # and it can perfectly well have a schedule where this one has
            # none. Marking it infeasible here would be writing down a
            # verdict nobody computed - and that verdict would be the
            # interesting one, because a bound that delivers where the
            # controller cannot is exactly what the bound is for.
            for code in missing:
                if code == "M1":
                    continue
                variants[code] = {
                    "status": "infeasible",
                    "feasible": False,
                    "why": str(error),
                }
            print(
                f"  Q={fraction:>5.0%}  no feasible schedule on the substituted "
                f"programme",
                flush=True,
            )
            # The certificate explains *why* there is no schedule, and it
            # is a second set of linear programmes with the same right to
            # come back without a verdict as the first. The verdict that
            # matters - no schedule - is already recorded; failing to
            # explain it must not throw away the point that has it.
            try:
                record["certificate"] = describe_certificate(
                    certify(programme, None)
                )
            except SolverUndecided as trouble:
                print(
                    f"  Q={fraction:>5.0%}  certificate NOT WRITTEN - the "
                    f"solver returned without deciding ({trouble}). The point "
                    f"has no schedule; what is missing is the reason.",
                    flush=True,
                )
            empty = True
            if "M1" not in missing:
                return record
        if "B1" in missing and not empty and not undecided:
            variants["B1"] = describe(first, names, time.perf_counter() - clock)
            report(fraction, "B1", variants["B1"])

    def attempt(code: str, work) -> None:
        """Solve one variant, and keep the three outcomes apart.

        Solved, infeasible, and "the solver was told to stop" are three
        different things. The last one is recorded as itself and never as
        the second: a point marked infeasible is a claim about the canal,
        and a solver that ran out of time has made no claim at all.
        """
        clock = time.perf_counter()
        try:
            answer = work()
        except SolverUndecided as error:
            elapsed = time.perf_counter() - clock
            variants[code] = {
                "status": "undecided",
                "feasible": None,
                "solver_status": getattr(error, "status", -1),
                "why": str(error),
                "seconds": round(elapsed, 3),
            }
            print(
                f"  Q={fraction:>5.0%}  {code} NOT SOLVED - the solver stopped on "
                f"a limit after {elapsed:.0f}s. This is not infeasibility.",
                flush=True,
            )
            return
        except (BaselineError, LeximinError) as error:
            variants[code] = {"status": "infeasible", "feasible": False, "why": str(error)}
            print(f"  Q={fraction:>5.0%}  {code} infeasible: {error}", flush=True)
            return
        variants[code] = describe(answer, names, time.perf_counter() - clock)
        report(fraction, code, variants[code])

    if "B2" in missing and not empty and not undecided:
        attempt("B2", lambda: time_shift(programme, first))
    if "B3" in missing and not empty:
        attempt("B3", lambda: utilitarian(programme))
    if "B4" in missing and not empty:
        attempt("B4", lambda: leximin(programme))
    if "B5" in missing and not empty:
        attempt("B5", lambda: min_spread(programme, start=first.z if first else None))
    if "M1" in missing:
        if bound_level_only:
            clock = time.perf_counter()
            try:
                entry = upper_bound_level(
                    programme, plant, method=bound_method, options=bound_options
                )
            except SolverUndecided as error:
                elapsed = time.perf_counter() - clock
                variants["M1"] = {
                    "status": "undecided",
                    "feasible": None,
                    "solver_status": getattr(error, "status", -1),
                    "why": str(error),
                    "seconds": round(elapsed, 3),
                }
                print(
                    f"  Q={fraction:>5.0%}  M1 NOT SOLVED - the solver "
                    f"returned without deciding after {elapsed:.0f}s. "
                    f"This is not infeasibility.",
                    flush=True,
                )
            except (BaselineError, LeximinError) as error:
                variants["M1"] = {
                    "status": "infeasible", "feasible": False, "why": str(error)
                }
                print(f"  Q={fraction:>5.0%}  M1 infeasible: {error}", flush=True)
            else:
                entry["seconds"] = round(time.perf_counter() - clock, 3)
                entry["feasible"] = True
                variants["M1"] = plain(entry)
                print(
                    f"  Q={fraction:>5.0%}  M1  worst {entry['worst']:.4f}  "
                    f"(level only)  {entry['seconds']:.1f}s",
                    flush=True,
                )
        else:
            attempt(
                "M1",
                lambda: upper_bound(
                    programme, plant, method=bound_method, options=bound_options
                ),
            )

    # --- the certificate, and the two-run check on the tie-break ----------
    if "certificate" not in record and not empty:
        clock = time.perf_counter()
        result = None
        written = True
        try:
            result = solve_leximin(programme.ratio)
        except SolverUndecided as error:
            # A certificate is a claim about the canal, and it reads the
            # same whether the programme has no schedule or the solver
            # merely failed to find out. Without a verdict there is no
            # claim to make, so none is written and the point stays
            # unfinished until some run gets an answer out of it.
            written = False
            print(
                f"  Q={fraction:>5.0%}  certificate NOT WRITTEN - the solver "
                f"returned without deciding ({error}). This is not "
                f"infeasibility.",
                flush=True,
            )
        except LeximinError:
            result = None
        if written:
            certificate = certify(programme, result)
            record["certificate"] = describe_certificate(certificate)
            record["certificate"]["seconds"] = round(time.perf_counter() - clock, 3)
            print(
                f"  Q={fraction:>5.0%}  certificate  "
                f"{'filled' if certificate.fulfilled else 'NOT filled'}  "
                f"{certificate.digest[:12]}",
                flush=True,
            )

    if "determinism" not in record and variants.get("B4", {}).get("feasible"):
        try:
            again = leximin(programme)
            twice = leximin(programme)
        except SolverUndecided as error:
            print(
                f"  Q={fraction:>5.0%}  determinism not measured - the solver "
                f"returned without deciding ({error})",
                flush=True,
            )
            return record
        record["determinism"] = {
            "l2_between_runs": float(np.linalg.norm(again.z - twice.z)),
            "tie_break_cost": plain(again.detail.get("tie_break_cost")),
            "same_ratios": bool(np.allclose(again.ratios, twice.ratios, atol=1e-12)),
        }
        print(
            f"  Q={fraction:>5.0%}  two runs differ by "
            f"{record['determinism']['l2_between_runs']:.3e}",
            flush=True,
        )
    return record


def report(fraction: float, code: str, entry: dict) -> None:
    if not entry.get("feasible"):
        return
    print(
        f"  Q={fraction:>5.0%}  {code}  worst {entry['worst']:.4f}  "
        f"total {entry['total']:.4f}  spread {entry['spread']:.4f}  "
        f"{entry['seconds']:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                "" if item is None else
                (f"{item:.10g}" if isinstance(item, float) else str(item))
                for item in row
            )
        )
    write_text(path, "\n".join(lines))


def build_tables(records: list[dict], label: str) -> list[Path]:
    tables = repo_root() / "results" / "tables"
    summary, ratios, certificates = [], [], []

    for record in sorted(records, key=lambda item: -item["fraction"]):
        percent = round(100 * record["fraction"])
        for code in CODES:
            entry = record.get("variants", {}).get(code)
            if entry is None:
                continue
            status = entry.get("status", "solved" if entry.get("feasible") else "infeasible")
            if status == "level only":
                summary.append(
                    [percent, code, status, entry["worst"], None, None,
                     entry.get("n_programmes"), entry.get("seconds")]
                )
                continue
            if status != "solved":
                summary.append(
                    [percent, code, status, None, None, None, None,
                     entry.get("seconds")]
                )
                continue
            summary.append(
                [
                    percent,
                    code,
                    status,
                    entry["worst"],
                    entry["total"],
                    entry["spread"],
                    entry["n_programmes"],
                    entry["seconds"],
                ]
            )
            for user, value in entry["ratios"].items():
                ratios.append([percent, code, user, value])

        certificate = record.get("certificate")
        if certificate is not None:
            worst_user, worst_value = "", None
            if certificate["ratios"]:
                worst_user = min(certificate["ratios"], key=certificate["ratios"].get)
                worst_value = certificate["ratios"][worst_user]
            certificates.append(
                [
                    percent,
                    1 if certificate["fulfilled"] else 0,
                    len(certificate["shortfalls_m3"]),
                    worst_user,
                    worst_value,
                    " ".join(certificate["impossible_users"]),
                    " ".join(certificate["relaxation"]["binding_families"]),
                    certificate["digest"],
                ]
            )

    written = [
        (
            tables / f"{label}_summary.csv",
            ["percent", "code", "status", "worst", "total", "spread",
             "programmes", "seconds"],
            summary,
        ),
        (
            tables / f"{label}_ratios.csv",
            ["percent", "code", "user", "ratio"],
            ratios,
        ),
        (
            tables / f"{label}_certificates.csv",
            ["percent", "fulfilled", "short", "worst_user", "worst_ratio",
             "impossible", "binding", "digest"],
            certificates,
        ),
    ]
    for path, header, rows in written:
        write_csv(path, header, rows)
    return [path for path, _, _ in written]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default="main", help="output name (default: main)")
    parser.add_argument(
        "--skip", action="append", default=[], metavar="CODE",
        help="leave a variant out; --skip M1 turns hours into minutes",
    )
    parser.add_argument(
        "--only", nargs="+", type=float, metavar="F",
        help="run only these fractions, e.g. --only 1.0 0.7 0.5",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="ignore what earlier runs wrote and solve everything again",
    )
    parser.add_argument("--filter-order", type=int, default=FILTER_ORDER)
    parser.add_argument("--cutoff", type=float, default=CUTOFF_RAD_PER_S)
    parser.add_argument("--overshoot", type=float, default=OVERSHOOT)
    parser.add_argument(
        "--time-limit", type=float, default=None, metavar="SECONDS",
        help=(
            "stop the free-gate bound's solver after this many seconds per "
            "programme. A stop is recorded as not solved, never as infeasible"
        ),
    )
    parser.add_argument(
        "--bound-level-only", action="store_true",
        help=(
            "solve only the free-gate bound's first lexicographic stage - the "
            "highest the worst-off user could have been - instead of the whole "
            "vector. One programme instead of nine, and it is the number the "
            "paper quotes; the record says which was computed"
        ),
    )
    parser.add_argument(
        "--bound-tolerance", type=float, default=None, metavar="EPS",
        help=(
            "primal and dual feasibility tolerance for the free-gate bound "
            "only. The frozen 1e-9 is very tight for a programme whose level "
            "recursion carries values across 240 steps, and a tolerance that "
            "tight can leave the simplex refining a number nobody reads: the "
            "fractions are reported to four decimals"
        ),
    )
    parser.add_argument(
        "--bound-method", default=None, choices=("highs", "highs-ds", "highs-ipm"),
        help=(
            "solver for the free-gate bound only; the frozen configuration is "
            "used for everything else, and whichever is used is recorded"
        ),
    )
    parser.add_argument(
        "--margin", type=int, default=None, metavar="STEPS",
        help=(
            "settling steps after the last order block (default: the frozen "
            "120). A filter whose memory is longer than the margin cannot be "
            "run at all, so a sensitivity run that lowers the cut-off needs "
            "this raised; the value used is recorded with the results"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    unknown = set(args.skip) - set(CODES)
    if unknown:
        print(f"unknown variant to skip: {sorted(unknown)}", file=sys.stderr)
        return 2
    wanted = tuple(code for code in CODES if code not in args.skip)
    fractions = tuple(args.only) if args.only else SCAN

    changed = [
        f"filter order {args.filter_order}" if args.filter_order != FILTER_ORDER else "",
        "" if args.bound_tolerance is None else "",  # bound settings are M1-only
        f"cut-off {args.cutoff} rad/s" if args.cutoff != CUTOFF_RAD_PER_S else "",
        f"overshoot {args.overshoot}" if args.overshoot != OVERSHOOT else "",
    ]
    changed = [item for item in changed if item]
    if changed:
        print(
            "NOT the pre-registered run: " + ", ".join(changed)
            + "\nthe pre-registration fixes these; this is a sensitivity "
              "run and has to be reported as one.\n",
            flush=True,
        )

    print(
        f"building the canal (filter order {args.filter_order}, "
        f"cut-off {args.cutoff} rad/s)...",
        flush=True,
    )
    clock = time.perf_counter()
    network, plant, mapping = build(args.filter_order, args.cutoff, args.margin)
    limits = make_limits(network)
    print(
        f"  {network.size} reaches, {mapping.steps} steps, "
        f"{BLOCKS} blocks, {time.perf_counter() - clock:.1f}s",
        flush=True,
    )
    bound_options = None
    if args.time_limit is not None or args.bound_tolerance is not None:
        bound_options = dict(LP_OPTIONS)
    if args.time_limit is not None:
        if args.time_limit <= 0.0:
            print("a time limit must be positive", file=sys.stderr)
            return 2
        bound_options["time_limit"] = args.time_limit
    if args.bound_tolerance is not None:
        if not 0.0 < args.bound_tolerance < 1.0:
            print("a tolerance must lie strictly between zero and one", file=sys.stderr)
            return 2
        bound_options["primal_feasibility_tolerance"] = args.bound_tolerance
        bound_options["dual_feasibility_tolerance"] = args.bound_tolerance

    print(
        f"scan: {len(fractions)} points, variants {' '.join(wanted)}",
        flush=True,
    )
    if "M1" in wanted and (bound_options or args.bound_method or args.bound_level_only):
        print(
            f"  bound solver: {args.bound_method or LP_METHOD}"
            + (f", {args.time_limit:g}s per programme" if args.time_limit else "")
            + (f", tolerance {args.bound_tolerance:g}" if args.bound_tolerance else "")
            + (", first stage only" if args.bound_level_only else "")
            + "\n  (the frozen configuration still applies to every other "
              "variant; this is recorded with the results)",
            flush=True,
        )
    print("", flush=True)

    out_dir = repo_root() / "results" / "scan" / args.label
    started = time.perf_counter()

    for fraction in fractions:
        percent = round(100 * fraction)
        path = out_dir / f"Q{percent:03d}.json"
        record = {}
        if path.is_file() and not args.fresh:
            record = json.loads(path.read_text(encoding="utf-8"))
        print(f"Q = {fraction:.0%}", flush=True)
        record = run_point(
            fraction,
            network,
            plant,
            mapping,
            limits,
            wanted,
            record,
            args.overshoot,
            bound_method=args.bound_method,
            settle_margin=args.margin or SETTLE_MARGIN_STEPS,
            bound_options=bound_options,
            bound_level_only=args.bound_level_only,
        )
        record["label"] = args.label
        record["filter_order"] = args.filter_order
        record["cutoff_rad_per_s"] = args.cutoff
        record["overshoot"] = args.overshoot
        record["bound_method"] = args.bound_method or LP_METHOD
        record["bound_time_limit_s"] = args.time_limit
        record["bound_tolerance"] = args.bound_tolerance
        record["settle_margin_steps"] = args.margin or SETTLE_MARGIN_STEPS
        write_json(path, record)

    # Tables are rebuilt from every point on disk, not only the ones this
    # run touched, so an interrupted scan still produces complete files.
    on_disk = []
    for path in sorted(out_dir.glob("Q*.json")):
        on_disk.append(json.loads(path.read_text(encoding="utf-8")))
    written = build_tables(on_disk, args.label)

    print(f"\nfinished in {time.perf_counter() - started:.0f}s", flush=True)
    print(f"  points   {out_dir.relative_to(repo_root()).as_posix()}", flush=True)
    for path in written:
        print(f"  table    {path.relative_to(repo_root()).as_posix()}", flush=True)
    print(
        "\nremember to record the environment that produced these:\n"
        "  python scripts/record_environment.py",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
