# RQ1a secondary-comparison reporting correction

This is an exploratory reporting correction computed from the supplied, saved paired-comparison table. It uses no new data, model calls, or experiment runs. It does not alter any frozen source, configuration, result file, confidence interval, non-inferiority result, or primary H1a decision.

## Files and reproduction

- `rq1a_holm_corrected.csv`: all 864 original rows and original columns, with the former `p_holm` column renamed `p_holm_original`; adds `p_holm_corrected` and `holm_family_size`.
- `rq1a_holm_verification.json`: verification checks, input/output SHA-256 hashes, family details, and rows whose threshold classification changes.
- `recompute_rq1a_holm.py`: Compatibility entry point for the Python standard-library release tool. From the repository root, run `python release_tools/recompute_reporting_corrections.py`.

Input: `adaptive-data-quality/results/v2/rq1_thresholds/test/rq1a_paired_ci.csv`.

Regenerated CSV and verification JSON are written to **.artifacts/reporting-corrections/**. The release tool checks that its CSV matches the archived corrected table in this directory. It never overwrites the archived result CSV, archived verification report, raw comparison table, or primary decision file. No research-data companion archive or third-party Python package is required.

## Declared family

Each family is one **dataset × scenario × metric** combination, across all distinct method/comparator pairs. There are 54 families (3 datasets × 9 scenarios × 2 metrics), each with 16 comparator configurations. All rows compare adaptive with another configuration. Distinct configurations remain distinct hypotheses even when they produce identical predictions.

For sorted raw p-values `p_(1) ≤ ... ≤ p_(m)`, the corrected value at rank `i` is the maximum, over ranks `j ≤ i`, of `min(1, (m-j+1) p_(j))`. Undefined raw p-values are assigned 1 solely for conservative family construction. The 48 undefined values occur in zero-error-control recall comparisons; an adjusted value of 1 there is bookkeeping, not a meaningful recall test.

The fixed-pooled comparison is included in this 16-comparison reporting family, but the original primary H1a decisions continue to use the prespecified FPR and recall confidence-interval conditions. Those decisions are unchanged.

## What changed

- 122 adjusted values changed, across 32 families.
- At strict `p < 0.05`, 568 original rows were below the threshold; 540 corrected rows are below it. **28 comparisons no longer meet this threshold.**
- At strict `p < 0.01`, the corresponding counts are 538 and 522. **16 comparisons no longer meet this threshold.**
- All six verification checks pass. Original input fields, row order and frozen primary decision-file hashes are preserved.

The original RQ1a runner applied Holm inside a separate two-method call for each comparator, leaving only one p-value in each correction family. This reanalysis applies the adjustment after collecting the comparator results.

## Limits of the correction

The original raw p-values are approximate two-sided bootstrap sign-tail proportions, rather than exact randomisation-test p-values. This script retains them as supplied; correcting multiplicity does not improve their underlying calibration. The 497 stored zero p-values remain zero after adjustment. Report such values as limited by the 5,000-resample resolution, not as an exact probability of zero.

This family definition corrects across comparators **within** one dataset, scenario and metric. It does not control a study-wide family spanning scenarios, datasets, metrics, experiments, or post-hoc analyses. Consequently, the corrected table cannot establish global superiority across the complete evaluation. Describe it as a secondary reporting correction and preserve the distinction between the original frozen decisions and this retrospective multiplicity repair.
