# Contributing

Reproducibility fixes, clearer documentation and well-scoped implementation improvements are welcome.

## Preserve the reported study

This release supports a specific manuscript and internally frozen evaluation. Do not overwrite archived results or change frozen settings merely to make a new run match the paper. Keep new runs separate, recording settings and environment.

Source changes can invalidate cached keys. Describe whether a pull request changes scientific behaviour, packaging or reporting. Keep reporting corrections separate from original measurements.

## Validation

Run from the repository root:

    python release_tools/verify_release.py

Then from adaptive-data-quality:

    python -m pytest -q -p no:cacheprovider

Explain the concrete problem, resulting behaviour, validation and effects on fingerprints, results or paper claims. Add meaningful tests for scientific changes; routine CI must not require paid services.

## Reports

Include release/commit, Python and package versions, operating system, command and redacted error. For numerical differences, identify dataset, split, seed, scenario, configuration and metric.

Never post credentials, bearer tokens, private service responses or unnecessary personal records. Report sensitive credential issues directly to the maintainers.

## Licensing and authorship

Original-software/documentation contributions use the MIT license. Preserve third-party notices; the root license does not relicense datasets or templates.

Authorship, contribution statements and scientific conclusions require author review. A code contribution does not automatically imply paper authorship.

