# DATA

Normalised inputs for the study, together with everything needed to rebuild
them from their published sources.

## Rules

* Every layer records the source URL, the access date, the licence and the
  exact query or export used, in `provenance.json`.
* `SHA256SUMS.txt` lists a checksum for every file here, so the build is its
  own regression test.
* This directory is produced by `scripts/build_data.py`. It is not edited by
  hand. Deleting it and re-running that script must write the same bytes
  back.
* No confidential or person-level data is used anywhere in this study.

Nothing has been built yet.
