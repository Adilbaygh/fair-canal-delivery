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

Everything the article reports is produced by the scripts below and written
to `results/`. Nothing is typed by hand.

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
the scripts below and written to `results/` or `DATA/`.

| Object | Produced by | Written to |
|---|---|---|
| Environment record | `scripts/record_environment.py` | `results/environment.json` |
| Model inputs, with provenance and citations | `scripts/build_data.py` | `DATA/` |
| Scarcity scan and sensitivity runs | `scripts/run_experiments.py` | `results/scan/<label>/` |
| Result tables | `scripts/make_tables.py` | `results/tables/` |
| Figures | `scripts/make_figures.py` | `results/figures/` |
| Interactive appendix | `scripts/make_explorer.py` | `results/explorer.html` |
| Journal readiness check | `scripts/check_submission.py` | printed, not stored |

## Reproducing everything

In this order; the whole sequence takes about twenty minutes, and the scan
is nearly all of it.

```
python scripts/record_environment.py
python scripts/build_data.py
python scripts/run_experiments.py --fresh --bound-level-only \
    --bound-tolerance 1e-9 --time-limit 300
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/make_explorer.py
python scripts/check_submission.py
```

The scan writes each supply level the moment that level finishes, so a run
stopped half way leaves a usable archive and the tables rebuild from
whatever points are present. The last command ends with a single line
saying whether every machine-checkable requirement is met.

## Determinism

The same inputs produce byte-identical outputs. Three rules make that true
and none of them is optional:

1. Anything whose order is incidental is sorted before it is written.
2. The solver, its version and its options are frozen in
   `src/faircanal/config.py`; they are recorded in `results/environment.json`
   on every run.
3. All text output is written with `"\n"` pinned and JSON keys sorted.

Determinism was measured, not assumed: at every solved supply level two
independent runs of the tie-break stage agree to `0.000e+00` in the
Euclidean norm, and the value is stored in each result file. One limit is
stated rather than hidden — the claim holds **on one machine**. The `B1`
baseline is an L1 projection whose optimal face is multi-vertex at tight
points, so it can land on a different vertex at the same L1 distance on a
different platform; `B4` and `B5` agree across platforms exactly.

## Layout

```
src/faircanal/     the library
src/faircanal/gui/ the results viewer, and nothing that computes
scripts/           computation scripts
DATA/              model inputs, provenance and citations
results/           result JSON, tables, figures, interactive appendix
tests/             permanent tests, never deleted
main.py            entry point
```

## Data

`DATA/` holds every input the model takes, exported from the source so the
canal behind `results/` can be read without opening a Python module. Each
block carries one of three provenance labels, and they are not
interchangeable: **observed** (published by someone else, with the citation
travelling beside the numbers), **derived** (computed from something
observed by a stated formula), **assumed** (chosen by this study because no
published value was found). The canal geometry comes from Bonet et al.
(2025), *Water* **17**(9):1368, `doi:10.3390/w17091368`, CC BY, with target
depths from Litrico & Fromion (2004). No confidential or person-level data
is used anywhere in this study.

The export is a copy, and a copy that can go stale is worse than none, so
`tests/test_data_export.py` rebuilds every exported value from the live
objects and fails naming the field that drifted.

## Licence

The **code** — everything under `src/`, `scripts/`, `tests/` and `main.py` —
is released under the MIT licence; see `LICENSE`.

The **data and results** — everything under `DATA/` and `results/` — are
released under Creative Commons Attribution 4.0 International
(CC BY 4.0); see `LICENSE-DATA`. That is the licence of the published
canal geometry this study builds on, so the attribution chain is kept
intact: reuse of these files should cite both this package and Bonet et al.
(2025).

## Citation

The machine-readable form is `CITATION.cff`. In prose:

> Kudaybergenov, A., Kazimbetova, M., Ametova, G., Ispanova, J., Absametov, B.,
> Qudaynazarov, M., Shikhiyev, R., Urazimbetova, E., & Nurullaev, Z. (2026).
> *Fair-Canal-Delivery: lexicographic max-min water delivery under deadline and
> capacity limits* (Version 0.2.0) [Computer software]. Zenodo.
> https://doi.org/10.5281/zenodo.22772981

Zenodo mints two DOIs and they are not interchangeable:

| DOI | Points at |
|---|---|
| `10.5281/zenodo.22772981` | **this version, 0.2.0** - cite this to name the exact code a number came from |
| `10.5281/zenodo.22723842` | all versions - resolves to whichever is newest |
| `10.5281/zenodo.22723843` | version 0.1.0, superseded - see the note below |

The archived file is `fair-canal-delivery-0.2.0.zip`, built with `git archive`
from the `v0.2.0` tag. No file can carry its own checksum, so the MD5 is
published in the two places outside it: beside the file on Zenodo, and in this
README on GitHub, added in the commit that follows the tag. Rebuild the archive
from the tag and compare it against either:

```
git archive --format=zip --prefix=fair-canal-delivery-0.2.0/ \
    -o fair-canal-delivery-0.2.0.zip v0.2.0
```

**Version 0.1.0 is superseded and should not be cited for a number.** Its
lexicographic solver contained a fault: the saturation test that decides when a
user can no longer be raised dropped a constant term from the fraction it was
testing, so on this study's programme every user was declared saturated at the
first stage and the procedure returned max-min where it promised leximin. The
worst-off delivered fraction - every headline number - is unaffected, because it
is the first stage's own answer; the fractions above it are not. The same release
corrects which programme's shadow prices the infeasibility certificate reports.
Both faults are covered by regression tests, and the whole result archive in 0.2.0
was re-solved by one version of the code.

The archive built from the 0.2.0 tag is md5:eb26477035449abf98b6822b8632b140.

The Zenodo **record** and both DOIs are public. The **files** are restricted
while the article is under review, and open on acceptance.
