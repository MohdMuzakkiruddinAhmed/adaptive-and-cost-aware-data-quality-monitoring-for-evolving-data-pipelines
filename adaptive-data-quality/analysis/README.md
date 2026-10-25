# Post-freeze analyses (added 20 Oct 2026)

These scripts run AFTER the confirmatory test run (protocol v2.2, frozen 20 Oct 2026). They import the frozen
pipeline unchanged. This folder is outside the folders hashed by `common/provenance.py` (CODE_DIRS) and outside
`configs/`, so adding or editing files here leaves the code fingerprint of the test run (`fd749de031ca4f54`) and
the configuration hashes unchanged: the cached test results stay valid and every new manifest still shows the
frozen pipeline fingerprint. Each script records its own file hash in its outputs.

| Script | What it fixes | Command | Outputs |
|---|---|---|---|
| `sensitivity.py` (+ `sensitivity.yaml`) | The Section 14 sensitivity analysis had only been run on the proof-of-concept pipeline (`experiments/sensitivity.py`). This one re-runs the frozen v2.2 monitor, one setting at a time, on the ten test seeds. | `python -m analysis.sensitivity --split test --jobs 4` | `results/v2/sensitivity/test/` |
| `chicago_adjudication.py` (+ `facility_type_suggestions.csv`) | Facility Type alarms of the Chicago replay had no verdict (free text, no documented domain) and the adjudication sample was never filled. Values are adjudicated once in a codebook; every alarm and sample row inherits the verdict of its value. | `python -m analysis.chicago_adjudication prepare`, fill the codebook, then `... apply` (or `apply --provisional`) | `results/v2/temporal_chicago/adjudication/` |
| `figures.py` | The v2 runners write tables only, so `results/figures/` stayed empty. Draws the paper figures from the test tables. | `python -m analysis.figures` | `results/figures/v2/test/` and `../paper/figures/v2/` |

## Sensitivity analysis

Dials (one changed at a time; everything else frozen). Values marked * come from `configs/experiment.yaml`
"sensitivity", fixed before freezing; the others were added after freezing and must be reported as such.

| Dial | Values | Evaluated on |
|---|---|---|
| defect_rate* | 1 (default), 3, 5, 10, 20 % | H1a: A_stable, B_drift; adaptive, fixed_pooled, ewma, rolling_mad_cal |
| outlier_sigma* | 3-5, 4-12 (default), 5-8, 8-12 sd | H1a |
| range_k | 2.5, 3 (default), 3.5, 4 | H1a |
| history_window | 3, 5 (default), 8, 10 | H1a |
| calibration_z* | 2, 2.5, 3 (default), 3.5, 4 | H1b: A-D; proposed, gated_fixed_pooled |
| drift_alpha | 0.01, 0.025, 0.05 (default), 0.10, 0.20 | H1b |
| risk_weights* | each weight x0, x0.5, x2; equal weights | H3: C, D, H1-H4; risk vs run-all |
| risk_thresholds* | moderate, high, very high, one at a time | H3 |

Claims re-checked in every setting (`sens_conclusions.csv`, summary in `sens_conclusions_summary.csv`):
C1 adaptive FPR below fixed_pooled under drift (CI); C2 recall non-inferior (1-point margin); C3 adaptive FPR not
above the better calibrated rolling baseline; C4 proposed macro-F1 at least that of gated_fixed_pooled; C5 risk
selection useful in sustained degradation (C, D). `sens_seed_spread.csv` gives the seed-to-seed spread of the
default setting, the yardstick for judging every change. Default-setting rows reproduce the main experiments
exactly (checked on dev seed 42).

Runtime: about 1 to 2 hours with `--jobs 4` (BTS is the slowest dataset). Interrupted runs resume per
(dataset, seed) unit. `--summary-only` rebuilds the tables.

## Chicago adjudication

`prepare` replays the raw data through the frozen monitor exactly as `experiments/v2/temporal_chicago.py` does
(the alarm counts are checked against the frozen run: 3,238 + 38 Facility Type alarms) and writes
`facility_type_codebook.csv`: one row per distinct flagged value (433), with alarm counts, a deterministic rule
pre-label (`pre_label`, `pre_label_rule`), a separate model suggestion for values no rule decides
(`suggested_verdict`, `suggestion_reason`, `suggestion_by`), and empty human columns.

What a verdict means: does the value correctly name a type of establishment?
- `error`: misspelled, garbled, or not a facility type at all;
- `not_error`: a recognisable type, including rare ones and pure format variants (case, spacing, plural,
  a license code next to the name; flagged in `format_variant`);
- `unclear`: cannot be decided from the value.

Human step: open the codebook (sorted by alarms), fill `adjudicator_1_verdict` (and `adjudicator_1`) for each
value; a second person can fill `adjudicator_2_verdict` for agreement and Cohen's kappa (`agreement.csv`).
Disagreements become `unclear` unless `resolved_verdict` is filled. Re-running `prepare` keeps every human entry.
Only human verdicts may be called adjudicated in the paper; `apply --provisional` uses the rule pre-labels and
then the model suggestions where no human verdict exists yet, and labels them as such in `verdict_source`.

## Figures

`python -m analysis.figures` draws: H1a FPR over drift and paired differences, H1b macro-F1 and confusion
matrices, H2a semantic recall, H2b category guard, H3 cost versus detection and runtime by phase, ablation main
effects, Tax benchmark, Chicago monthly replay, Chicago Facility Type verdicts, and the sensitivity claims grid
(once the sensitivity analysis has run). A figure whose input table is missing is skipped with a note.
