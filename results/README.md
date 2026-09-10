# results

Everything the article reports is written here by a script, and nothing here
is edited by hand.

* `environment.json` - the machine, the operating system, the library
  versions and the frozen solver configuration of the run.
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
  five.
* `figures/` - one PDF and one EPS per reported figure, produced from the
  files above by `scripts/make_figures.py`.

A figure nobody can regenerate is a claim, not evidence, so the script that
draws each figure is part of the published set. The same goes for the
tables: they are derived from the scan files, never typed.
