#!/usr/bin/env python3
"""Does every recorded digest still say what its record was solved on?

A certificate's digest exists to answer one question - were these two
runs the same problem under the same settings? - so it has two ways to
fail, and both have happened in this study:

* two runs that differ in something that matters share a digest. The
  filter was missing from the payload once, and the gate limit once; each
  time the collision was found by comparing two runs that had given
  different answers, not by reading the code.
* a recorded digest no longer matches what the code computes, because the
  payload gained a field after the run was written.

This checks both across every point of every label in ``results/scan``.
It rebuilds each record's instance from the settings the record itself
carries - through the same ``scenario_for`` the scan uses, so there is
one definition of the instance and not two - recomputes the digest, and
reports:

* records whose stored digest differs from the recomputed one;
* digests shared by records that were not solved on the same settings.

With ``--fix`` it writes the recomputed digest back into the record (and
into the certificate's own report line, which quotes it), leaving every
other field untouched. Nothing else in a record depends on the digest, so
this is a rewrite of one derived field and not a re-run: what it writes
is what a fresh scan would write, which is what the collision check then
confirms.

    python scripts/check_digests.py          # report only
    python scripts/check_digests.py --fix    # report and rewrite
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from faircanal.certificate import _digest  # noqa: E402
from faircanal.config import (  # noqa: E402
    ANNOUNCE_BLOCKS,
    BAND_TOLERANCE_M,
    GATE_HEAD_M,
    OUTLET_HEADROOM,
    SETTLE_MARGIN_STEPS,
)
from faircanal.programme import assemble  # noqa: E402
from faircanal.provenance import repo_root, write_json  # noqa: E402
from run_experiments import (  # noqa: E402
    build,
    build_tables,
    make_limits,
    scenario_for,
)


def settings_of(record: dict) -> dict:
    """What this record says it was solved on, with the frozen defaults."""
    return {
        "filter_order": record.get("filter_order", 3),
        "cutoff": record.get("cutoff_rad_per_s", 3.0e-3),
        "margin": record.get("settle_margin_steps", SETTLE_MARGIN_STEPS),
        "gate_head": record.get("gate_head_m", GATE_HEAD_M),
        "cap_scale": record.get("cap_scale", 1.0),
        "band": record.get("band_tolerance_m", BAND_TOLERANCE_M),
        "announce": record.get("announce_block", ANNOUNCE_BLOCKS),
        "headroom": record.get("outlet_headroom", OUTLET_HEADROOM),
        "demand_scale": record.get("demand_scale", 1.0),
        "overshoot": record.get("overshoot", 0.0),
    }


def recompute(record: dict, cache: dict) -> str:
    s = settings_of(record)
    key = (s["filter_order"], s["cutoff"], s["margin"])
    if key not in cache:
        cache[key] = build(s["filter_order"], s["cutoff"], s["margin"])
    network, _plant, mapping = cache[key]
    limits = make_limits(
        network,
        gate_head_m=s["gate_head"],
        cap_scale=s["cap_scale"],
        band_tolerance_m=s["band"],
    )
    scenario = scenario_for(
        record["fraction"],
        network,
        limits,
        overshoot=s["overshoot"],
        settle_margin=s["margin"],
        announced_block=s["announce"],
        outlet_headroom=s["headroom"],
        demand_scale=s["demand_scale"],
    )
    return _digest(assemble(scenario, mapping))


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)

    scan = repo_root() / "results" / "scan"
    cache: dict = {}
    stale, seen = [], defaultdict(list)
    checked = 0

    for path in sorted(scan.glob("*/Q*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        certificate = record.get("certificate")
        if not certificate:
            continue
        name = f"{path.parent.name}/{path.stem}"
        fresh = recompute(record, cache)
        checked += 1
        stored = certificate.get("digest", "")
        if stored != fresh:
            stale.append((name, stored, fresh, path, record))
        seen[fresh].append((name, tuple(sorted(settings_of(record).items())),
                            record["fraction"]))

    print(f"{checked} records checked in {scan.relative_to(repo_root()).as_posix()}\n")

    if stale:
        print(f"{len(stale)} record(s) whose stored digest is not what the code "
              f"computes now:")
        for name, stored, fresh, _path, _record in stale:
            print(f"  {name:<18} stored {stored[:12]}  now {fresh[:12]}")
    else:
        print("every stored digest matches the code.")

    collisions = {
        digest: rows
        for digest, rows in seen.items()
        if len({(settings, fraction) for _n, settings, fraction in rows} ) > 1
    }
    print()
    if collisions:
        print(f"{len(collisions)} digest(s) shared by runs that were NOT the same "
              f"problem:")
        for digest, rows in collisions.items():
            print(f"  {digest[:12]}: " + ", ".join(name for name, _s, _f in rows))
    else:
        print("no digest is shared by two runs with different settings.")

    if stale and args.fix:
        print()
        touched = set()
        for name, stored, fresh, path, record in stale:
            certificate = record["certificate"]
            certificate["digest"] = fresh
            report = certificate.get("report")
            if report and stored[:12] in report:
                certificate["report"] = report.replace(stored[:12], fresh[:12])
            write_json(path, record)
            touched.add(path.parent)
            print(f"  rewritten {name}")
        # The tables quote the digest too, and a table that disagreed with
        # the record it was built from would be a third place for the same
        # number to be wrong in.
        for folder in sorted(touched):
            records = [
                json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(folder.glob("Q*.json"))
            ]
            for written in build_tables(records, folder.name):
                print(f"  rebuilt   {written.relative_to(repo_root()).as_posix()}")
        print(f"\n{len(stale)} record(s) rewritten. Run this again without --fix "
              f"to confirm.")
        return 0

    return 1 if (stale or collisions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
