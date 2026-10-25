"""Post-freeze analyses of the frozen v2.2 pipeline (sensitivity analysis, Chicago adjudication, figures).

This folder is deliberately outside common/provenance.py CODE_DIRS and configs/: adding or changing files here
does not change the code fingerprint or the configuration hashes of the frozen protocol (test run 20 Oct 2026,
code fingerprint fd749de031ca4f54). Every script here imports the frozen pipeline unchanged and records its own
file hash in its outputs.
"""
