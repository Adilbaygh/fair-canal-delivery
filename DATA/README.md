# DATA

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
