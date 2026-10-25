"""Run the complete v2 evaluation (all seven supervisor experiments) in a sensible order.

    python -m experiments.v2.run_all --split dev --jobs 4                 # exploratory / debugging runs
    python -m experiments.v2.run_all --split test --jobs 4                # confirmatory: needs protocol.frozen: true

Order
  0  materialize          ground truth, labels, audits, profiles (reproducibility)
  0a export_tools         replay batches for the established tools (Deequ, TFDV)
  0b tools                run the Deequ and TFDV sidecars (Linux / WSL; failures are reported, not fatal:
                          the experiments then skip those comparators with a warning)
  1  rq1_thresholds       adaptive thresholds vs strong baselines (H1a)
  2  rq1_classification   drift vs degradation, drift-capable comparators, gate ablation (H1b)
  3  rq2_coverage         validated LLM rules vs statistical and manual rules (H2a)     [needs LLM generations]
  4  rq2_category_guard   guarded category admission (H2b)                              [manual part needs none]
  4a validator_stress     deliberately broken rules through the admission checks (v2.1)
  5  rq3_scheduling       schedulers with measured cost (H3) (+ --runtime trials, single process)
  6  ablation             8 component combinations in one loop
  7  external             dirty/clean benchmarks (needs scripts/download_benchmarks.py); single- and cross-column rules
  8  temporal_chicago     chronological raw Chicago replay (no labels; objective verdicts + adjudication sample)

LLM generations are NOT made here (they cost money): run python -m experiments.v2.llm_generate first, or pass
--with-llm-generation (rules for the three datasets and the external benchmarks, and the per-value checks of the
category guard; cached answers are reused at no cost). Steps that need them warn and skip LLM configurations when they are missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.v2 import (ablation, external, llm_generate, materialize, rq1_classification,  # noqa: E402
                            rq1_thresholds, rq2_category_guard, rq2_coverage, rq3_scheduling, run_tools,
                            temporal_chicago, tool_exchange, validator_stress)

STEPS = ["materialize", "export_tools", "tools", "rq1_thresholds", "rq1_classification", "rq2_coverage", "rq2_category_guard",
         "validator_stress", "rq3_scheduling", "ablation", "external", "temporal_chicago"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--steps", nargs="*", default=STEPS, choices=STEPS)
    ap.add_argument("--with-llm-generation", action="store_true", help="make (paid) LLM generations first")
    ap.add_argument("--runtime", action="store_true", help="also the timed runtime trials of rq3")
    ap.add_argument("--force", action="store_true", help="recompute everything (cached results are otherwise "
                                                          "reused only while code and configuration are unchanged)")
    args = ap.parse_args()
    common = ["--split", args.split] + (["--force"] if args.force else [])
    par = common + ["--jobs", str(args.jobs)]
    if args.with_llm_generation:
        llm_generate.main(["--split", args.split])
    t0 = time.perf_counter()
    for step in args.steps:
        print(f"\n=== {step} ({args.split}) ===", flush=True)
        if step == "materialize":
            materialize.main(par)
        elif step == "export_tools":
            tool_exchange.main(common)
        elif step == "tools":
            run_tools.main(["--split", args.split] + (["--force"] if args.force else []))
        elif step == "rq1_thresholds":
            rq1_thresholds.main(par)
        elif step == "rq1_classification":
            rq1_classification.main(par)
        elif step == "rq2_coverage":
            rq2_coverage.main(par)
        elif step == "rq2_category_guard":
            rq2_category_guard.main(par)
        elif step == "validator_stress":
            validator_stress.main(par)
        elif step == "rq3_scheduling":
            rq3_scheduling.main(par)
            if args.runtime:
                rq3_scheduling.main(common + ["--runtime", "--summary-only"])
        elif step == "ablation":
            ablation.main(par)
        elif step == "external":
            external.main(common)
        elif step == "temporal_chicago":
            temporal_chicago.main([])
    print(f"\nAll steps done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
