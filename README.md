# Fair-Canal-Delivery

Reproducibility package for the study *Deadline-Constrained Lexicographic
Max-Min Water Delivery in Dynamically Limited Irrigation Canal Networks*.

The study asks a single question: when an irrigation canal cannot serve every
order in full, **who gets short-changed, and by how much?** Fairness is
measured on the volume a user actually receives inside an agreed delivery
window, not on the volume that was scheduled. Orders are reshaped *before*
the low-pass filter that suppresses wave dynamics; the existing structured LQ
controller and the filter itself are left untouched. When an order cannot be
met, the method returns an infeasibility certificate instead of an optimistic
schedule.

**Status: under construction.** The model is fixed and verified; the
implementation is being built step by step. Only the objects listed in
"Reported objects" below are currently produced.

## Requirements

* Python >= 3.11
* The pinned dependency set in `requirements-lock.txt`

## Install

```
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
pip install -e .
```

## Run

```
python main.py env                # record the environment
python main.py check              # run the permanent test suite
pytest                            # the same suite, directly
```

## Reported objects

Every number, table and figure reported in the article is produced by one of
the scripts below and written to `results/`. Nothing is typed by hand.

| Object | Produced by | Written to |
|---|---|---|
| Environment record | `scripts/record_environment.py` | `results/environment.json` |
| Normalised benchmark inputs | `scripts/build_data.py` | `DATA/` |
| Experiment results | `scripts/run_experiments.py` | `results/` |
| Figures | `scripts/make_figures.py` | `results/figures/` |

Rows are added as each step lands. A row is only added once the script
exists and has been run.

## Determinism

The same inputs produce byte-identical outputs. Three rules make that true
and none of them is optional:

1. Anything whose order is incidental is sorted before it is written.
2. The solver, its version and its options are frozen in
   `src/faircanal/config.py`; they are recorded in `results/environment.json`
   on every run.
3. All text output is written with `"\n"` pinned and JSON keys sorted.

## Layout

```
src/faircanal/     the library
scripts/           computation scripts
DATA/              benchmark inputs, provenance and checksums
results/           result JSON, tables, figures
tests/             permanent tests, never deleted
main.py            entry point
```

## Data

`DATA/` contains normalised inputs derived from published, openly licensed
sources, together with a provenance record naming each source, its licence
and the access date, and SHA-256 checksums for every file. No confidential
or person-level data is used anywhere in this study.

## Citation

See `CITATION.cff`.
