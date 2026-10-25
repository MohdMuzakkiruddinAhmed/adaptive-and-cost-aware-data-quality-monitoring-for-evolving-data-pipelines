"""Rerun everything with one command (Section 21):

    python -m experiments.run_all                   # prepare data, Experiments 1, 2, 4, 5
    python -m experiments.run_all --with-llm        # ... and Experiments 3 and 3b (Bedrock; cached answers cost nothing)
    python -m experiments.run_all --skip-prepare    # reuse existing batches
    python -m experiments.run_all --sensitivity     # ... and the Section 14 sensitivity analysis (~17 min)

Experiment 5 includes its LLM configurations whenever Experiment 3 results exist.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import (exp1_static_vs_adaptive, exp2_drift_vs_degradation, exp3_llm_rules,  # noqa: E402
                         exp3b_semantic_defects, exp4_cost_optimization, exp5_ablation, prepare_data,
                         sensitivity)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--with-llm", action="store_true", help="also run Experiment 3 (AWS Bedrock)")
    ap.add_argument("--sensitivity", action="store_true", help="also run the Section 14 sensitivity analysis")
    args, rest = ap.parse_known_args()
    prog = sys.argv[0]
    if not args.skip_prepare:
        sys.argv = [prog] + rest
        prepare_data.main()
    sys.argv = [prog]
    print("\n=== Experiment 1: static vs adaptive ===")
    exp1_static_vs_adaptive.main()
    print("\n=== Experiment 2: drift vs degradation ===")
    exp2_drift_vs_degradation.main()
    if args.with_llm:
        print("\n=== Experiment 3: LLM rule generation ===")
        exp3_llm_rules.main()
        print("\n=== Experiment 3b: LLM rules on semantic defects ===")
        exp3b_semantic_defects.main()
    print("\n=== Experiment 4: cost-aware execution ===")
    exp4_cost_optimization.main()
    print("\n=== Experiment 5: ablation ===")
    exp5_ablation.main()
    if args.sensitivity:
        print("\n=== Section 14: sensitivity analysis ===")
        sensitivity.main()
