# results

Everything the article reports is written here by a script, and nothing here
is edited by hand.

* `environment.json` - the machine, the operating system, the library
  versions, the frozen solver configuration, and the commit that produced
  the run.
* `scan/<label>/Q<pct>.json` - one file per scarcity point, written by
  `scripts/run_experiments.py` the moment that point finishes. Everything
  the pre-registration said would be reported is in it: the fractions of
  every variant, the lexicographic stage levels and who saturated at each,
  the structural ceilings, the binding constraint families with their
  shadow prices, the elastic relaxation in its own units, the certificate
  and its digest, and how long each solve took. `main` is the
  pre-registered scan; any other label is a sensitivity run and says in
  its own files what was changed.
* `tables/` - one CSV per reported table, rebuilt from the files above on
  every run, so they are complete whether the scan ran in one sitting or
  five. The `.md` copies carry the reported numbers in the wording the
  article uses.
* `figures/` - one PNG and one PDF per reported figure, produced from the
  files above by `scripts/make_figures.py`, with `captions.md` beside them.
  The **PNG at 600 dpi is what the journal receives**; the PDF is a vector
  master kept so a figure can be re-exported without redrawing it. Widths
  are the journal template's own: 184.6 mm for a figure that spans the text
  block, 138.6 mm for one that sits in the column.
* `explorer.html` - the interactive appendix the article points at. One
  self-contained file, no network, so it works from a downloaded copy.

A figure nobody can regenerate is a claim, not evidence, so the script that
draws each figure is part of the published set. The same goes for the
tables: they are derived from the scan files, never typed.

## What a missing number means here

Three verdicts are kept apart and never folded into one another. A point is
**solved** (a schedule exists and the numbers come from it), **infeasible**
(no schedule exists, and the solver proved it), or **undecided** (the solver
returned no verdict at all). The third is not a rounding of the second: it
means nobody computed the value. A field that has no computed number is
written as `null`, never as a zero and never as `NaN` — every file here is
strict JSON and a strict reader will accept it.

## Licence

The files in this directory are released under Creative Commons Attribution
4.0 International (CC BY 4.0); see `LICENSE-DATA` in the repository root.
