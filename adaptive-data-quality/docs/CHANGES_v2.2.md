# Protocol v2.2: final changes after development run 4 (frozen for testing)

All changes use development data only (dev pool, dev seeds 42 to 46). The test pool was never used before freezing.
Development run 5 is the complete run with v2.2; the protocol was frozen after it.

## Changes

**1. Drift gate with selective execution** (`monitor/classifier.py drift_evidence`)

- Problem (a bug introduced by v2.1 change 2): the `gated_executed` drift source reuses only executed checks that
  match a gate rule exactly. Since v2.1 the monitored category check also keeps the warm-up categories, while the
  gate keeps the recurrence-only rule, so the two never matched and the whole category gate was skipped, even with
  every check executed. Invalid categories entered the drift scores, and degradation batches were labelled
  drift + degradation. Affected: the ablation and RQ3 (both use `gated_executed`); RQ1b, RQ2b and the Chicago replay
  use `gated` and were not affected.
- Change: a gate rule whose check (same type and column) was executed this batch is computed in the gate's own form.
  Checks the scheduler skipped still give no gate evidence, as designed.
- Effect (dev run 4): gate coverage with the full suite back to 1.0 (was 0.77 to 0.84); ablation macro-F1 of the
  adaptive configurations BTS 0.498 -> 0.680, Chicago 0.513 -> 0.649, Retail 0.573 -> 0.674. Detection and every
  RQ3 table unchanged.

**2. Manual rules pass the same admission test as the LLM rules** (`configs/manual_rules.yaml validate_manual: true`,
`rules/manual_rules.py admitted_manual_rules`, `experiments/v2/rq2_coverage.py`, `experiments/v2/rq2_category_guard.py`)

- Problem: the independent author (Llama 4 Maverick, data dictionary only) made factual errors that no admission
  step caught (the `validate_manual` flag existed but was never read):
  - BTS `Reporting_Airline` pattern `^[A-Z]{2}$` rejects real carrier codes B6, F9, 9E, G4 (11.0% of clean records);
  - Chicago `Results` list omits Out of Business, No Entry, Not Ready (12.9%);
  - Chicago `Facility Type` pattern rejects apostrophes and brackets (3.9%); Retail `StockCode` pattern rejects
    POST, DOT, M, 15056BL (0.9%).
  As a result the manual condition flagged 10.8% (BTS) and 22.3% (Chicago) of valid records in dev run 4.
- Change: in every run the manual rules go through `rules/validator.py validate` on that run's own warm-up batches
  (logical redundancy mode), exactly as the LLM rules do. A practitioner tests an expert rule against history before
  deploying it, and the comparison of manual and LLM rules is then symmetric. The rules as written are still
  reported: `manual_raw`, `fixed+manual_raw`, `adaptive+manual_raw` (RQ2a) and `recurrence_manual_raw` (RQ2b).
  Admission outcomes of the manual rules appear in `rq2a_admission_errors.csv` (model `manual`).
- Effect (dev seed 42 warm-up): the four faulty rules above are rejected (patterns) or relaxed (Results list gains
  the warm-up outcomes); every other manual rule is admitted unchanged.

**3. Admission outcomes judge a relaxed rule in its relaxed form** (`evaluation/rule_analysis.py admission_outcomes`)

- Problem: a candidate that admission relaxed ("modified") was scored on the evaluation batches as written, not as
  executed. Dev run 5: the relaxed Chicago Results list (warm-up outcomes added) raised almost no false alarms, but
  was counted as "incorrectly admitted" because the list as written flags 13 to 19% of valid records.
- Change: modified rules are scored in the form that is executed. Rejected and redundant candidates are still
  scored as written. No LLM rule was modified in any dev run, so the LLM rows are unchanged.

**Dev run 5 (complete, v2.2; 2 h 4 min, all manifests share one code fingerprint).** Manual rules after admission:
false-positive rate 0.000% (BTS), 0.025% (Chicago), 0.000% (Retail), against 10.8%, 22.3% and 0.9% as written;
semantic recall 0.50 / 0.67 / 0.45 (as written 0.53 / 0.73 / 0.54); 0 incorrectly admitted or rejected manual rules.
The manual category guard becomes weaker on BTS and Retail (the rejected carrier and stock-code patterns no longer
constrain those columns; invalid novelty accepted in 100% of cases). Everything else as in dev run 4 after the gate fix.

## Decisions (no change)

- **Author model of the manual rules: unchanged (Llama 4 Maverick).** Grok (xAI) was considered. It is not offered
  on this Bedrock account (probe 19 Oct 2026) and no xAI credentials are configured. Replacing the author after
  seeing its development results would also be a post-hoc choice. With change 2 the author's slips are filtered by
  the same test as the LLM's. The rules remain a model's rules, not an expert's: say so in the paper.
- **Weak or mixed results are reported, not tuned:** Retail H1a not supported (FPR equal to pooled; recall CI below
  the margin under drift); the thin Chicago H1a margin (both methods use the city box); low recall on the
  contamination scenarios (the Chicago one is undetectable inside the city); Chicago drift recall 0.82; H3 useful in
  11 of 24 cases (weak on sparse duplicates and short incidents). Changing methods or scenarios now to make these
  pass would tune toward the hypotheses.
- **Tax single-column cell recall 0.289 -> 0.201 (run 3 -> run 4) is a definition change, not a regression:** since
  v2.1 the denominator is all error cells in the checkable columns (monitored + context columns,
  `experiments/v2/external.py _cell_counts`), not only the monitored columns.
- **Decision criteria kept as prespecified:** recall non-inferiority margin 1 point, FPR increase tolerance 0.5
  points, detection target 0.90. They were set before any test data was used and are not changed.
- **Human tasks that do not block testing:** independent human review of the manual rules (the rules stay
  model-authored), adjudication of the Chicago Facility Type sample (`temporal_adjudication.csv`; an evaluation of the
  replay output, done after the run).
