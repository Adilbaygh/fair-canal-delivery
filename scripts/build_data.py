#!/usr/bin/env python3
"""Write out every input the model takes, with where each one came from.

The results were reproducible from the day the archive was written, but
only to someone willing to read Python: the canal's geometry, the filter,
the limits and the frozen constants all live inside the source, and
``DATA/`` held nothing but a note. That is a real gap in a reproducibility
package. A reader should be able to see what canal produced these numbers
without opening a module.

So this exports them. Two files, for two readers:

``DATA/corning_canal.csv``
    The published canal as a table - one row per reach, the columns a
    hydraulic engineer would expect. Opens in anything.
``DATA/inputs.json``
    Everything, including what a CSV cannot carry: the citation behind
    each block, the provenance label (observed, derived, assumed) and the
    constants the pre-registration froze.

Exported, not authoritative
---------------------------
The code remains the source of truth; these files are written from it,
not read back into it. An exported copy that can drift from the code it
describes is worse than no copy, so ``tests/test_data_export.py`` rebuilds
every value from the live objects and fails if the files disagree. Run
this after any change to the canal, the filter or the frozen constants.

    python scripts/build_data.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from faircanal.benchmarks import haughton_filter  # noqa: E402
from faircanal.config import (  # noqa: E402
    D_MIN_M3,
    DT_BLOCK_S,
    DT_PLANT_S,
    EPS_SAT,
    LP_METHOD,
    LP_OPTIONS,
    R_CAP,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.delivery import memory_steps  # noqa: E402
from faircanal.geometry import uniform_discharge  # noqa: E402
from faircanal.network import corning_cascade  # noqa: E402
from faircanal.provenance import repo_root, write_json, write_text  # noqa: E402

# The pre-registration's own constants, named here exactly as the scan
# names them. If these two lists ever disagree the export is lying, which
# is what the test is for.
BLOCKS = 8
LEAD_BLOCKS = 2
WARM_UP_STEPS = 15
TRAVEL_FRACTION = 0.25
FILTER_ORDER = 3
CUTOFF_RAD_PER_S = 3.0e-3
OVERSHOOT = 0.0
SCAN = tuple(round(1.00 - 0.05 * step, 2) for step in range(15))

CSV_HEADER = [
    "node", "label", "length_m", "bed_width_m", "side_slope", "canal_depth_m",
    "manning_n", "bed_slope", "target_level_m", "offtake_m3_s", "tail_m3_s",
    "capacity_m3_s", "level_band_low_m", "level_band_high_m",
    "travel_rate_m3_s",
]


def canal_rows() -> list[dict]:
    """One row per reach: what is published, and what follows from it."""
    network = corning_cascade()
    rows = []
    for reach in network.reaches:
        pool = reach.pool
        full = uniform_discharge(pool, pool.canal_depth_m)
        band = pool.canal_depth_m - pool.target_level_m
        rows.append({
            "node": reach.node,
            "label": reach.label,
            "length_m": pool.length_m,
            "bed_width_m": pool.bed_width_m,
            "side_slope": pool.side_slope,
            "canal_depth_m": pool.canal_depth_m,
            "manning_n": pool.manning_n,
            "bed_slope": pool.bed_slope,
            "target_level_m": pool.target_level_m,
            "offtake_m3_s": reach.offtake_m3_s,
            "tail_m3_s": reach.tail_m3_s,
            "capacity_m3_s": full,
            "level_band_low_m": -band,
            "level_band_high_m": band,
            "travel_rate_m3_s": TRAVEL_FRACTION * full,
        })
    return rows


def payload() -> dict:
    """Everything, each block with its provenance and its citation."""
    network = corning_cascade()
    pool = network.reaches[0].pool
    spec = haughton_filter(FILTER_ORDER)
    return {
        "canal": {
            "name": network.name,
            "reaches": len(network.reaches),
            "topology_provenance": network.topology_provenance,
            "topology_rule": network.topology_rule,
            "source": network.source,
            "geometry_provenance": pool.provenance,
            "geometry_source": pool.source,
            "steady_discharges_m3_s": list(network.steady_discharges),
            "aggregate_demand_m3_s": network.aggregate_demand,
            "table": canal_rows(),
        },
        "limits": {
            "provenance": (
                "capacity derived from the geometry at full supply level; "
                "level band derived from canal depth minus target level; "
                "gate travel rate and warm-up assumed"
            ),
            "travel_rate_fraction_of_capacity": TRAVEL_FRACTION,
            "warm_up_steps": WARM_UP_STEPS,
        },
        "filter": {
            "provenance": "assumed - the wave-damping filter of the source study",
            "kind": "Butterworth low-pass",
            "order": spec.order,
            "cutoff_rad_per_s": spec.cutoff_rad_per_s,
            "sample_time_s": spec.sample_time_s,
            "memory_steps": memory_steps(spec),
        },
        "discretisation": {
            "dt_plant_s": DT_PLANT_S,
            "dt_block_s": DT_BLOCK_S,
            "steps_per_block": STEPS_PER_BLOCK,
            "settle_margin_steps": SETTLE_MARGIN_STEPS,
            "horizon_steps": BLOCKS * STEPS_PER_BLOCK + SETTLE_MARGIN_STEPS,
        },
        "scenario": {
            "provenance": "assumed - stated in the pre-registration, section 1.5",
            "blocks": BLOCKS,
            "lead_blocks": LEAD_BLOCKS,
            "overshoot": OVERSHOOT,
            "ratio_cap": R_CAP,
            "min_demand_m3": D_MIN_M3,
            "scan_fractions": list(SCAN),
        },
        "solver": {
            "method": LP_METHOD,
            "options": dict(LP_OPTIONS),
            "saturation_tolerance": EPS_SAT,
            "note": (
                "The solver removes matrix coefficients below its own "
                "threshold, which cannot be set through SciPy, so the "
                "programme solved is not exactly the programme assembled. "
                "Measured: removing them changes no reported fraction to ten "
                "decimal places; the study's quality audit records the "
                "measurement and the limitation together."
            ),
        },
        "note": (
            "Exported from the source by scripts/build_data.py. The code is "
            "the source of truth; these files are written from it and are "
            "checked against it by tests/test_data_export.py."
        ),
    }


README = """# DATA

Every input the model takes, exported from the source so that the canal
behind `results/` can be read without opening a Python module.

| File | What it holds |
|---|---|
| `corning_canal.csv` | The published canal as a table, one row per reach |
| `inputs.json` | Everything, with each block's citation and provenance label |

## Provenance, in three words

Each block in `inputs.json` carries one of three labels, and they are not
interchangeable:

- **observed** - published by someone else and cited. The canal geometry,
  the offtakes and the check flows are all observed, and the citation is
  in the file.
- **derived** - computed from something observed by a stated formula. The
  conveyance capacity of each reach is the uniform discharge at full
  supply level; the level band is the canal depth minus the target level.
- **assumed** - chosen by this study because no published value was found.
  The gate travel rate, the warm-up, the delivery window and the filter
  are assumed, and every result that depends on them says so.

## These files are exported, not read

The code remains authoritative. `scripts/build_data.py` writes these from
the live objects and `tests/test_data_export.py` rebuilds every value and
fails if they disagree, so an export cannot quietly drift from the model
it claims to describe.

Regenerate with:

    python scripts/build_data.py

## Licence and attribution

These files are released under Creative Commons Attribution 4.0
International (CC BY 4.0); the statement is in `LICENSE-DATA` in the
repository root. CC BY is also the licence of the published canal geometry
this study builds on, so the chain is kept intact rather than relicensed:
anyone reusing these files should cite this package **and** the source of
the geometry, which `inputs.json` names in full beside the numbers it
applies to.
"""


def main() -> int:
    folder = repo_root() / "DATA"
    folder.mkdir(parents=True, exist_ok=True)

    rows = canal_rows()
    lines = [",".join(CSV_HEADER)]
    for row in rows:
        lines.append(",".join(_cell(row[name]) for name in CSV_HEADER))
    written = [
        write_text(folder / "corning_canal.csv", "\n".join(lines)),
        write_json(folder / "inputs.json", payload()),
        write_text(folder / "README.md", README),
    ]
    for path in written:
        print(f"  {path.relative_to(repo_root())}")
    return 0


def _cell(value) -> str:
    if isinstance(value, float):
        # Enough digits that a reader can rebuild the model exactly, and
        # not so many that the table becomes unreadable.
        return f"{value:.10g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
