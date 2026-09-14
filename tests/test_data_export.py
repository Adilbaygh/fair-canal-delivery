"""Permanent tests for the exported inputs. Never deleted.

`DATA/` holds a copy of every input the model takes, so that the canal
behind the results can be read without opening a Python module. A copy is
useful exactly as long as it is true, and a copy of numbers that live
somewhere else goes stale the first time somebody changes the original
and forgets the export.

So these tests rebuild every exported value from the live objects and
compare. If the canal changes, or the filter, or a frozen constant, and
`scripts/build_data.py` has not been rerun, the suite fails and names the
field that drifted. That is the whole point: the export is allowed to be
a copy because it cannot be a stale one.

The other half is provenance. Every published number in this study is
labelled observed, derived or assumed, and the label is what lets a
reader tell "this is what the canal is" from "this is what we chose". A
block that lost its label would look exactly like one that never needed
one, so each block is checked for having its own.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from faircanal.benchmarks import haughton_filter
from faircanal.config import (
    ANNOUNCE_BLOCKS,
    BAND_TOLERANCE_M,
    DT_BLOCK_S,
    DT_PLANT_S,
    GATE_HEAD_M,
    OUTLET_HEADROOM,
    R_CAP,
    SETTLE_MARGIN_STEPS,
    STEPS_PER_BLOCK,
)
from faircanal.delivery import memory_steps
from faircanal.geometry import corning_gate_limits, uniform_discharge
from faircanal.network import corning_cascade
from faircanal.provenance import repo_root

DATA = repo_root() / "DATA"
CSV = DATA / "corning_canal.csv"
JSON = DATA / "inputs.json"
TOL = 1.0e-9


def rebuild() -> None:
    """Regenerate the export, so a failure is about drift and not absence."""
    script = repo_root() / "scripts" / "build_data.py"
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)


@pytest.fixture(scope="module", autouse=True)
def exported():
    if not (CSV.exists() and JSON.exists()):
        rebuild()


def payload() -> dict:
    return json.loads(JSON.read_text(encoding="utf-8"))


def rows() -> list[dict]:
    with CSV.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# The export is the model, or it is a lie
# ---------------------------------------------------------------------------


def test_every_reach_in_the_table_is_the_reach_in_the_model():
    """Geometry, offtake and tail flow, reach by reach.

    Read from the file and compared to the object the scan actually
    builds. A change to the canal that does not reach the table fails
    here rather than in a reader's spreadsheet.
    """
    network = corning_cascade()
    table = rows()
    assert len(table) == len(network.reaches)
    for reach, row in zip(network.reaches, table):
        pool = reach.pool
        assert int(row["node"]) == reach.node
        assert row["label"] == reach.label
        assert float(row["length_m"]) == pytest.approx(pool.length_m, abs=TOL)
        assert float(row["bed_width_m"]) == pytest.approx(pool.bed_width_m, abs=TOL)
        assert float(row["side_slope"]) == pytest.approx(pool.side_slope, abs=TOL)
        assert float(row["canal_depth_m"]) == pytest.approx(pool.canal_depth_m, abs=TOL)
        assert float(row["manning_n"]) == pytest.approx(pool.manning_n, abs=TOL)
        assert float(row["bed_slope"]) == pytest.approx(pool.bed_slope, abs=TOL)
        assert float(row["target_level_m"]) == pytest.approx(
            pool.target_level_m, abs=TOL
        )
        assert float(row["offtake_m3_s"]) == pytest.approx(reach.offtake_m3_s, abs=TOL)
        assert float(row["tail_m3_s"]) == pytest.approx(reach.tail_m3_s, abs=TOL)


def test_the_derived_columns_are_derived_the_way_the_model_derives_them():
    """Capacity, level band and travel rate are not observations.

    They follow from the geometry by a stated rule, and the table has to
    apply the same rule the scan does - otherwise a reader recomputing
    from the CSV gets a different canal than the one that produced the
    results.
    """
    network = corning_cascade()
    for reach, row in zip(network.reaches, rows()):
        pool = reach.pool
        full = uniform_discharge(pool, pool.canal_depth_m)
        band = pool.canal_depth_m - pool.target_level_m
        assert float(row["capacity_m3_s"]) == pytest.approx(full, rel=1e-9)
        assert float(row["level_band_high_m"]) == pytest.approx(band, abs=TOL)
        assert float(row["level_band_low_m"]) == pytest.approx(-band, abs=TOL)
        assert float(row["travel_rate_m3_s"]) == pytest.approx(0.25 * full, rel=1e-9)


def test_the_gate_the_export_names_is_the_gate_the_programme_uses():
    """The narrower of the two limits has to be in the published table.

    On seven of this canal's eight reaches the check gate passes less than
    the reach conveys, so a table that listed only the conveyance would
    describe a canal whose gates are wider than the ones in the source's
    own table - and a reader recomputing the results from it would get a
    more generous canal than the one that produced them.
    """
    limits = corning_gate_limits(GATE_HEAD_M)
    table = rows()
    assert len(limits) == len(table)
    narrower = 0
    for gate, row in zip(limits, table):
        cell = row["gate_capacity_m3_s"]
        if cell == "":
            # The head of the canal: a heading structure, no published gate.
            assert gate == float("inf")
            continue
        assert float(cell) == pytest.approx(gate, rel=1e-9)
        if gate < float(row["capacity_m3_s"]):
            narrower += 1
    assert narrower == 7, (
        f"the gate binds on {narrower} reaches rather than seven - either the "
        f"head or the gate table has moved, and the article says seven"
    )
    block = payload()["limits"]
    assert block["gate_head_m"] == pytest.approx(GATE_HEAD_M, abs=TOL)
    assert block["storage_band_m"] == pytest.approx(BAND_TOLERANCE_M, abs=TOL)
    assert "measured" in block["storage_band_provenance"]


def test_the_scenario_block_carries_the_settings_the_scan_runs_with():
    block = payload()["scenario"]
    assert block["announce_block"] == ANNOUNCE_BLOCKS
    assert block["outlet_headroom"] == pytest.approx(OUTLET_HEADROOM, abs=TOL)
    assert block["announce_block"] < block["lead_blocks"], (
        "the shortage would be announced in the block it takes effect in"
    )


def test_the_filter_in_the_export_is_the_filter_in_the_study():
    spec = haughton_filter(3)
    block = payload()["filter"]
    assert block["order"] == spec.order
    assert block["cutoff_rad_per_s"] == pytest.approx(spec.cutoff_rad_per_s, abs=TOL)
    assert block["sample_time_s"] == pytest.approx(spec.sample_time_s, abs=TOL)
    assert block["memory_steps"] == memory_steps(spec)


def test_the_frozen_constants_are_the_frozen_constants():
    block = payload()["discretisation"]
    assert block["dt_plant_s"] == pytest.approx(DT_PLANT_S, abs=TOL)
    assert block["dt_block_s"] == pytest.approx(DT_BLOCK_S, abs=TOL)
    assert block["steps_per_block"] == STEPS_PER_BLOCK
    assert block["settle_margin_steps"] == SETTLE_MARGIN_STEPS
    assert payload()["scenario"]["ratio_cap"] == pytest.approx(R_CAP, abs=TOL)


def test_the_aggregate_demand_matches_the_network():
    network = corning_cascade()
    block = payload()["canal"]
    assert block["aggregate_demand_m3_s"] == pytest.approx(
        network.aggregate_demand, rel=1e-12
    )
    assert block["steady_discharges_m3_s"] == pytest.approx(
        list(network.steady_discharges), rel=1e-12
    )


def test_the_settling_margin_covers_the_exported_filter():
    """The one consistency the export could hide.

    A filter whose memory exceeds the margin cannot be run at all, and
    the export would happily record both numbers side by side without
    noticing. Checked here so that the exported configuration is a
    configuration that runs.
    """
    data = payload()
    assert data["filter"]["memory_steps"] <= data["discretisation"]["settle_margin_steps"]


# ---------------------------------------------------------------------------
# Provenance, which is what the labels are for
# ---------------------------------------------------------------------------


def test_every_block_says_where_its_numbers_came_from():
    """No block of inputs is allowed to arrive unlabelled."""
    data = payload()
    assert data["canal"]["geometry_provenance"]
    assert data["canal"]["geometry_source"]
    assert data["canal"]["topology_provenance"]
    assert data["canal"]["source"]
    for block in ("limits", "filter", "scenario"):
        assert data[block]["provenance"], f"{block} has no provenance label"


def test_the_published_geometry_carries_its_citation():
    """A DOI, not a hand-wave.

    The canal is somebody else's measurement and the export is where a
    reader looks for it, so the citation travels with the numbers rather
    than living only in the manuscript.
    """
    source = payload()["canal"]["geometry_source"]
    assert "doi" in source.lower()
    assert "10.3390/w17091368" in source


def test_the_solver_note_states_the_limitation_it_has_to_state():
    """The solver edits the matrix, and the export says so.

    This is the one place a reader rebuilding the model from DATA would
    otherwise be misled: the programme solved is not exactly the
    programme assembled.
    """
    note = payload()["solver"]["note"].lower()
    assert "threshold" in note and "not exactly" in note


# ---------------------------------------------------------------------------
# The export cannot be stale
# ---------------------------------------------------------------------------


def test_regenerating_changes_nothing(tmp_path):
    """Byte-identical on a second run, so a diff means a real change."""
    before = (CSV.read_bytes(), JSON.read_bytes())
    rebuild()
    after = (CSV.read_bytes(), JSON.read_bytes())
    assert before == after, (
        "DATA/ is out of date with the model - run scripts/build_data.py "
        "and commit the result"
    )
