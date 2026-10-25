# Implementation architecture

The framework processes recurring fixed-schema batches while keeping distribution change, constraints and prior failures separate.

## Batch order

1. Read and profile the batch.
2. Compute raw profile drift for scheduling.
3. Construct rules and adaptive bounds, including admitted candidates.
4. Select checks using current risk and prior observed outcomes.
5. Execute selected checks and collect available failure masks.
6. Form gated classification drift and failure evidence.
7. Assign stable, drift, degradation or drift-with-degradation.
8. Store profiles and outcomes.

Scheduling drift precedes execution; gated classification uses later evidence. Skipped checks supply no new failure observation. All profiles enter adaptation history; gating does not quarantine learning.

| Module | Responsibility |
|---|---|
| ingestion | Loading, cleaning, splitting, batching, benchmark label conventions |
| profiling | Current summaries and history |
| injection | Deterministic scenarios and audited labels |
| rules | Candidate generation, predicates, admission, cross-column rules |
| adaptation | Numerical policies and category guards |
| drift | Distribution distances |
| monitor | Integrated loop, classifier and schedulers |
| risk, optimizer | Scores and selection |
| execution | Checks, masks and logical work accounting |
| baselines | Historical comparator and Deequ/TFDV adapters |
| evaluation | Metrics and paired seed-cluster statistics |
| experiments/v2 | Frozen runners and per-unit caches |
| analysis | Post-freeze sensitivity, figures and adjudication |

The AVH-inspired comparator is a local historical-statistics implementation, not the full published optimiser. Deequ and TFDV use separate environments; archived outputs are restored from the research-assets archive.

## Cache boundaries

common/provenance.py lists the pipeline directories covered by the code fingerprint. Configuration hashes cover configs. Changes can invalidate cached units. Release documentation, manuscript prose and reporting repairs are kept outside these frozen inputs.

Cached model responses contain proposals rather than credentials. New model generation is optional and separate from routine tests. Installation and CI do not regenerate the scientific study.

