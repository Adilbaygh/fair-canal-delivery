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
python main.py                    # open the results viewer
python main.py env                # record the environment
python main.py check              # run the permanent test suite
pytest                            # the same suite, directly
```

## The results viewer

`python main.py` opens a desktop window: an appendix to the article that
shows every published result beside the file it came from, and runs the
published scripts on request. It reads `results/` and `DATA/` and computes
nothing of its own.

```
python -m pip install -r requirements-gui.txt   # PyQt6, once
python main.py                                  # open the window
python main.py --language en                    # open it in English
python main.py --list-pages                     # works without the toolkit
```

Two rules hold everywhere in it.

**A number that nobody computed is not displayed.** A supply level was
solved, or proved infeasible, or left without a verdict by the solver, and
the three are kept apart. Where there is no computed value there is a dash
— never a zero, never a blank cell.

**Every number carries its own file.** The grey line under a card or a
table is the file and key the figure was read from, so any figure on screen
can be checked against the archive.

The toolkit is optional and deliberately so: every number this study
reports is produced by the scripts, which do not need it, and the test
suite passes without it — the window's own tests skip. Pages: the claim,
the delivery window, the five criteria side by side at a supply level you
choose, the three verdicts and the infeasibility certificate, the two
theorems with a witness read live out of the archive, the sensitivity runs,
the inputs with their provenance, the figures and tables, and the buttons
that rebuild the lot.

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
src/faircanal/gui/ the results viewer, and nothing that computes
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
