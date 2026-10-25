# Paper-to-artifact experiment map

Paper: **Adaptive and Cost-Aware Data Quality Monitoring for Evolving Data Pipelines**.

Author order: Muzakkiruddin Ahmed Mohammed, Ahsan Shahid, Selimhan Dagtas, John Talburt, Abdelrahman Elfikky.

Use v2 test outputs for the paper's principal findings. Do not pool them with earlier proof-of-concept runs. Evidence paths below are relative to adaptive-data-quality.

| Component | Implementation | Evidence directory and tables |
|---|---|---|
| H1a thresholds | experiments/v2/rq1_thresholds.py | results/v2/rq1_thresholds/test/: rq1a_record_level.csv, rq1a_paired_ci.csv, rq1a_h1a_decisions.csv |
| H1b batch labels | experiments/v2/rq1_classification.py | results/v2/rq1_classification/test/: rq1b_classification_summary.csv, rq1b_confusion_matrices.csv, rq1b_paired_ci.csv |
| H2a semantic coverage | experiments/v2/rq2_coverage.py | results/v2/rq2_coverage/test/: rq2a_summary.csv, rq2a_paired_ci.csv, rq2a_admission_errors.csv |
| Admission stress | experiments/v2/validator_stress.py | results/v2/validator_stress/test/: validator_stress_summary.csv, validator_stress_status.csv |
| H2b category guard | experiments/v2/rq2_category_guard.py | results/v2/rq2_category_guard/test/: rq2b_summary.csv, rq2b_acceptance_per_value.csv |
| H3 scheduling | experiments/v2/rq3_scheduling.py | results/v2/rq3_scheduling/test/: rq3_detection_cost.csv, rq3_h3_decisions.csv, rq3_incidents_per_run.csv |
| Runtime | experiments/v2/rq3_scheduling.py | results/v2/rq3_scheduling/test/: rq3_runtime_trials_raw.csv, rq3_runtime.csv |
| Factorial ablation | experiments/v2/ablation.py | results/v2/ablation/test/: ablation_summary.csv, ablation_main_effects.csv |
| Tax/Beers transfer | experiments/v2/external.py | results/v2/external/test/: external_detection.csv, external_by_column.csv |
| Chicago chronology | experiments/v2/temporal_chicago.py | results/v2/temporal_chicago/: temporal_monthly.csv, temporal_change_point_check.csv |
| Sensitivity | analysis/sensitivity.py | results/v2/sensitivity/test/: sens_conclusions.csv, sens_conclusions_summary.csv |
| Tool comparators | experiments/v2/tool_exchange.py, run_tools.py | results/v2/tool_exchange, restored from companion assets |
| Legacy development | experiments/exp*.py | results/tables, results/raw, results/sensitivity |

## Shared design

- BTS flights, Online Retail II and Chicago food inspections.
- Twenty batches per run: five warm-up and fifteen evaluation batches.
- Disjoint 30% development and 70% test pools.
- Development seeds 42–46; paired test seeds 1001–1010.
- Seventeen scenarios covering drift, background/sustained degradation, mild errors, persistent contamination, impossible values, novelty and incidents.
- Protocol v2.2 and pipeline fingerprint fd749de031ca4f54.

Seeds describe simulation variation conditional on these source datasets, not independently sampled application domains. Controlled batches are shuffled; the Chicago replay uses natural monthly order.

## Interpretation boundaries

- High classification F1 does not establish numerical adaptation's advantage everywhere. Gated fixed thresholds give identical Chicago predictions and no detected Retail disadvantage.
- Semantic recall 0.996–1.00 belongs to fixed statistical plus admitted LLM rules. Adaptive combinations and the separate category guard are different configurations.
- The guard tests 24 distinct values, repeated across seeds.
- H3 usefulness uses positive saving and mean recall at least 0.90. It is not a lower-confidence-bound guarantee. C/D uses all post-onset errors; H1–H4 uses incident records during the incident.
- Tax's approximately 0.908 cell recall is within checkable columns; coverage of all erroneous cells is approximately 0.615. Cell precision must also be considered.
- The chronology contains a known July 2018 change, but lacks exhaustive event labels. Facility Type alarms await independent human adjudication.
- Stored runtime summaries total fifteen evaluation batches; the manuscript converts their units.

## Paper and figures

The LaTeX entry point is paper/latex/main.tex; the compiled document is paper/paper.pdf. Four figures show the monitoring loop, false positives under drift, category admission and chronology. Plot values and provenance accompany the manuscript's QA material.

The reporting-correction notes explain secondary Holm families and limitations. These retrospective corrections are not additional frozen experiments.

