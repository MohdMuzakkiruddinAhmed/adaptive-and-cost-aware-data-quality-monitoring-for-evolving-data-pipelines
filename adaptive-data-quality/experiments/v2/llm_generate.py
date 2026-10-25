"""Generate LLM rule proposals for the v2 experiments (AWS Bedrock). THIS SCRIPT CAN SPEND MONEY.

    python -m experiments.v2.llm_generate --dry-run          # prompts, token counts and cost estimate; no calls
    python -m experiments.v2.llm_generate --mock             # offline stand-in (pipeline test, $0)
    python -m experiments.v2.llm_generate                    # real calls for every model in protocol.yaml rq2.models

Covers the three study datasets, the external benchmarks (tax, beers; skipped when not downloaded; single-column
AND cross-column proposals, rules/llm_crosscolumn.py) and the per-value checks of the category guard
(experiments/v2/llm_value_checks.py).

Models: protocol.yaml llm_models (primary = the PoC model, secondary = a second model); generations per model
and dataset: protocol.yaml rq2.generations. Prices and the spending cap (llm.max_cost_usd, per process) are in
experiment.yaml; the run stops before any call that could exceed the cap. Answers are cached with a
fingerprint (experiments/v2/llm_rules.py); rerunning reuses them at no cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import datasets_config, experiment_config, protocol_config, v2_results_dir, write_json  # noqa: E402
from experiments.v2.llm_rules import (cache_dir, llm_settings, model_slug, models,  # noqa: E402
                                      prompt_and_fingerprint)
from injection.scenario import DataSource  # noqa: E402
from rules.llm_generator import BudgetExceeded, LLMClient  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, help="keys of protocol.yaml llm_models")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--no-benchmarks", action="store_true", help="skip the external benchmarks (tax, beers)")
    ap.add_argument("--no-value-checks", action="store_true", help="skip the per-value checks of the category guard")
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="split whose guard scenarios define the value-check candidates")
    args = ap.parse_args(argv)
    datasets, exp, pr = datasets_config(), experiment_config(), protocol_config()
    src = DataSource(datasets, exp, pr)
    from common.utils import data_dir, load_yaml
    from experiments.v2.llm_rules import benchmark_prompt_and_fingerprint
    bench = load_yaml("benchmarks")
    bench_names = [] if args.no_benchmarks or args.datasets else \
        [n for n, b in bench["benchmarks"].items()
         if not b.get("optional") and data_dir("external", b["files"]["dirty"]).exists()]
    gens = args.generations or pr["rq2"]["generations"]
    rows = []
    for key, model_id in models(pr, args.models).items():
        client = LLMClient(llm_settings(exp, model_id), mock=True if args.mock else None)
        for ds in (args.datasets or list(datasets)) + bench_names:
            pf = benchmark_prompt_and_fingerprint(ds, bench, exp, pr, model_id) if ds in bench_names else \
                prompt_and_fingerprint(src, ds, model_id)
            d = cache_dir(model_id, ds, pf["fingerprint"]["key"] + ("_mock" if client.mock else ""))
            n_in, how = client.count_input_tokens(pf["prompt"]) if not args.dry_run or client.mock else \
                (int(len(pf["prompt"]) / 3.2), "estimate")
            pdir = v2_results_dir("llm_prompts", model_slug(model_id))
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / f"{ds}.txt").write_text(pf["prompt"], encoding="utf-8")
            write_json(pf["fingerprint"], pdir / f"{ds}_fingerprint.json")
            cached = sum(1 for g in range(1, gens + 1) if (d / f"run_{g}.json").exists())
            row = {"model": key, "model_id": model_id, "dataset": ds, "fingerprint": pf["fingerprint"]["key"],
                   "input_tokens": n_in, "token_count": how, "generations": gens, "cached": cached,
                   "worst_case_usd": (gens - cached) * client.cost(n_in, int(client.cfg["max_tokens"]))}
            if not args.dry_run:
                try:
                    for g in range(1, gens + 1):
                        rec = client.generate(ds, g, pf["prompt"], n_in, cache_dir=d, fingerprint=pf["fingerprint"])
                        how_paid = "cache" if rec["from_cache"] else "$%.4f" % rec["cost_usd"]
                        print(f"[{key} {ds}] gen {g}: {len(rec['rules'] or [])} rules ({how_paid})", flush=True)
                except BudgetExceeded as exc:
                    print(f"STOPPED: {exc}")
                    rows.append(row)
                    break
            rows.append(row)
        if bench_names:                                 # cross-column proposals for the external benchmarks (v2.1)
            from experiments.v2.llm_rules import benchmark_xcol_prompt_and_fingerprint, xcol_cache_dir
            from rules.llm_crosscolumn import CrossColumnClient
            xclient = CrossColumnClient(llm_settings(exp, model_id), mock=client.mock)
            xclient.spent_usd = client.spent_usd             # one budget per model and process
            for ds in bench_names:
                pf = benchmark_xcol_prompt_and_fingerprint(ds, bench, exp, pr, model_id)
                d = xcol_cache_dir(model_id, ds, pf["fingerprint"]["key"] + ("_mock" if xclient.mock else ""))
                n_in, how = xclient.count_input_tokens(pf["prompt"]) if not args.dry_run or xclient.mock else \
                    (int(len(pf["prompt"]) / 3.2), "estimate")
                pdir = v2_results_dir("llm_prompts", model_slug(model_id))
                (pdir / f"{ds}__xcol.txt").write_text(pf["prompt"], encoding="utf-8")
                cached = sum(1 for g in range(1, gens + 1) if (d / f"run_{g}.json").exists())
                row = {"model": key, "model_id": model_id, "dataset": f"{ds}__xcol",
                       "fingerprint": pf["fingerprint"]["key"], "input_tokens": n_in, "token_count": how,
                       "generations": gens, "cached": cached,
                       "worst_case_usd": (gens - cached) * xclient.cost(n_in, int(xclient.cfg["max_tokens"]))}
                rows.append(row)
                if args.dry_run:
                    continue
                try:
                    for g in range(1, gens + 1):
                        rec = xclient.generate(f"{ds}__xcol", g, pf["prompt"], n_in, cache_dir=d,
                                               fingerprint=pf["fingerprint"])
                        how_paid = "cache" if rec["from_cache"] else "$%.4f" % rec["cost_usd"]
                        print(f"[{key} {ds} cross-column] gen {g}: {len(rec['rules'] or [])} rules ({how_paid})",
                              flush=True)
                except BudgetExceeded as exc:
                    print(f"STOPPED: {exc}")
                    break
            client.spent_usd, client.calls = xclient.spent_usd, client.calls + xclient.calls
        print(f"[{key}] new spend this run: ${client.spent_usd:.4f} over {client.calls} call(s)")
        if not args.no_value_checks and not args.datasets:
            from experiments.v2.llm_value_checks import run_value_checks
            run_value_checks(src, key, model_id, split=args.split, mock=client.mock, dry_run=args.dry_run)
    est = pd.DataFrame(rows)
    out = v2_results_dir("llm_prompts")
    est.to_csv(out / ("generation_estimate_mock.csv" if args.mock else "generation_estimate.csv"), index=False)
    pd.set_option("display.width", 200)
    print(est.to_string(index=False))
    if args.dry_run:
        print(f"\nDry run: no calls made. Prompts in {out}. Worst case for missing generations: "
              f"${est.worst_case_usd.sum():.2f}")


if __name__ == "__main__":
    main()
