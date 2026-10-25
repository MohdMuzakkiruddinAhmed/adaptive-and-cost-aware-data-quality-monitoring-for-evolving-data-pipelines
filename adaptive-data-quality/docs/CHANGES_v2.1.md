# Protocol v2.1: changes after development run 3

All changes were made on development data only (dev pool, dev seeds 42 to 46). The test pool stays locked until
`protocol.frozen: true`. Each change below states the problem seen in development run 3, the change, and the
effect measured on development data.

## Method

**1. Drift thresholds from resampled stable batches** (`drift/drift_detector.py resampled_null_scores`,
`monitor/classifier.py`; `protocol.yaml classification.drift_threshold.resampling`)

- Problem: 15 to 17% of stable batches on Chicago and Retail were labelled drift (BTS 4.7%). The Student-t
  prediction bound was built from five leave-one-out scores. On discrete, heavy-tailed columns (Retail Quantity and
  Price) these five scores were nearly equal, so the estimated spread was too small.
- Change: the warm-up records are split at random into five pseudo-batches, 200 times. Each pseudo-batch is scored
  against the other four, in the same way as the leave-one-out calibration. This gives 1,000 scores of a stable batch
  per column. The threshold is the larger of the leave-one-out bound and the (1 - alpha / C) quantile of these
  scores. The leave-one-out bound still covers real differences between batches.
- Effect (dev seeds, scenarios A to D, proposed classifier): false drift fell to 2.0% on BTS, 4.0% on Chicago and
  4.7% on Retail. Drift recall was unchanged on BTS and Retail (1.00). On Chicago it went from 0.93 to 0.88, and the
  only drifts missed were at the smallest levels (0.02 to 0.06). Degradation recall on Retail rose from 0.92 to 0.98.

**2. Warm-up categories stay valid** (`adaptation/adaptive_thresholds.py`; `experiment.yaml
adaptive.category_keep_warmup`)

- Problem: the recurrence rule accepted a category only if it appeared in at least two of the last five batches.
  A rare legitimate category that had already been seen in the warm-up was rejected again whenever it was missing
  for a while. As a result, every rolling method had a higher false-alarm rate than the pooled baseline on Retail
  (adaptive 2.5% against pooled 1.1%).
- Change: every category seen in the warm-up stays valid, and newly recurring categories are learned as before. The
  warm-up is trusted in the same way by the pooled and frozen baselines. The gate that filters errors out of the
  drift estimate keeps the old rule, because a looser gate at evaluation time than at calibration time raised false
  drift.
- Effect (seed 42): the false-alarm rate of the adaptive method on Retail fell from 2.5% to 1.1%, the same level as
  the pooled baseline.

**3. Coordinates are checked against the city, not a k-sigma band** (`configs/datasets.yaml
chicago_food.domain_ranges`, `monitor/suite.py`, `ingestion/geo.py`)

- Problem: in the chronological Chicago replay, most "degradation" months came from latitude and longitude range
  alarms. Of the coordinate alarms in the sample, 99.3% were inside the city limits. They were legitimate
  establishments at the edge of the city (O'Hare, the far South Side). A band around the mean position has no
  meaning for a location.
- Change: in every method, Chicago latitude and longitude are checked against the bounding box of the official city
  boundary (City of Chicago, "Boundaries - City", qqq8-j68g). The statistical range rule is not used for these two
  columns.
- Effect: the false-alarm rate on Chicago fell from about 2% to 0.4%. Coordinate outliers injected inside the city
  can no longer be detected (recall 0.99), and the Chicago contamination scenario (longitude x 1.002) becomes
  undetectable for all methods, because the shifted positions are still inside the city.

**4. Cross-column rules for the external benchmarks** (`rules/crosscolumn.py`, `rules/llm_crosscolumn.py`,
`experiments/v2/external.py`)

- Problem: on Tax, most remaining errors only show up in combination with another field (a zip code from another
  place, a "no children" flag on a record with a child exemption, "married" with a single exemption). Single-column
  rules cannot detect them.
- Change: two new rule types. A dependency (for example zip determines city) uses a mapping learned from the
  warm-up. A conditional constraint has the form "if has_child = N then child_exemp = 0". These rules come from two
  sources. They are discovered from the warm-up (`adaptive+deps`, no LLM), or the LLM proposes their structure
  (`adaptive+xcol_<model>`), in which case mappings are learned from the warm-up and the same admission test is
  applied. Benchmark columns that have no single-column checks (zip, city, area code) can be used in these rules
  (`context_columns`).
- Effect (Tax, seed 42, learned rules only): record-level recall rose from 0.16 to 0.63. Cell-level recall over the
  checkable columns rose from 0.20 to 0.92. The false-alarm rate went from 3.07% to 3.16%.

## Evaluation and comparators

**5. Deequ evaluated by Deequ itself** (`tools/external_validators/run_deequ.py`, `baselines/external_tools.py`)

- Before, Deequ's suggestions were translated into this project's rule engine. On a warm-up that still contained
  errors, Deequ suggested almost no record-level constraints.
- Now Deequ suggests the constraints, verifies them on every evaluation batch, and reports the violating records
  through its row-level results. It is run in two ways: as a practitioner would run it (warm-up as observed, DEFAULT
  rules) and in its best case (warm-up without its erroneous records, Deequ's EXTENDED rules without the NUMERICAL ones,
  which assert exact equality with the warm-up value and, for the mean, use the maximum: a Deequ bug). Its records are counted in two
  readings: violating a constraint that failed on the batch, or violating any constraint. The share of batches with a
  failed constraint is also reported.
- Batch level: Deequ's anomaly detection is reported as documented (OnlineNormalStrategy, cumulative history). It is
  also reported with the warm-up as reference and a deviation factor set by the same calibration as the proposed
  method, with or without failed suggested constraints as an extra degradation signal.
- A self-test (`run_deequ.py --selftest`) checks every Deequ call before a run.

**6. Benchmark prompts no longer contain evaluated rows** (`configs/benchmarks.yaml prompt_pool`,
`ingestion/benchmarks.py`)

- Ten percent of each benchmark table (at least 500 rows), chosen by the split seed, is reserved for the LLM prompts
  and never enters a replay batch. Before, the prompt sample overlapped the evaluation batches of other seeds.

**7. Objective verdicts in the Chicago replay** (`experiments/v2/temporal_chicago.py objective_verdicts`)

- Every alarm gets an objective verdict where one exists:
  - position inside or outside the city boundary polygon;
  - missing value;
  - exact duplicate;
  - value inside or outside the documented domain of Risk and Results.
- Only Facility Type alarms remain unresolved, so human adjudication is needed only for those.

**8. Validator stress test** (`experiments/v2/validator_stress.py`)

- Rules with known defects are generated from each run's warm-up: malformed, too strict, wrong, subtly too strict,
  sound, vacuous and redundant rules.
- They go through the same admission code as the LLM rules. Each candidate's false alarms are measured as written
  and as admitted.

**9. Independent manual rules** (`experiments/v2/manual_author.py`, `rules/llm_author.py`)

- The manual rules are rewritten by a model of a different family from the two rule-generating LLMs (Llama 4
  Maverick by default).
- The model sees only the data dictionary: no data, no statistics and no injected error types.
- The v2.0 draft is kept for reference.

**10. H1a verdict for the zero-defect control**

- When a scenario has no erroneous records, recall is undefined and the H1a verdict uses the false-alarm part only.
  Before, these scenarios were counted as "not supported".
