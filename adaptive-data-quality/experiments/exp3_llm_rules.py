"""Experiment 3: LLM Rule Generation (project description, Sections 8.3, 9, 13).

Research question  Does LLM-assisted rule generation improve useful rule coverage when
                   combined with validation?
Comparison         Static/manual rules  vs  LLM rules (as generated, executable)  vs
                   LLM + statistical validation (Section 9); plus static + validated LLM rules
                   to measure additional coverage.
Measures           Generated rules, valid/invalid, rejected, modified, redundant, coverage,
                   false alarms, defects detected (all four scenarios, T6..T20), stability
                   across repeated generations.

    python -m experiments.exp3_llm_rules --dry-run   # build prompts, count tokens, estimate cost; no LLM call
    python -m experiments.exp3_llm_rules             # generate (or reuse cached answers), validate, evaluate

Model, number of runs, prices and the spending cap are in configs/experiment.yaml (llm).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import batch_label, data_dir, datasets_config, experiment_config, results_dir, write_json  # noqa: E402
from experiments.cost_aware import run  # noqa: E402
from experiments.detection import load_batch  # noqa: E402
from experiments.exp4_cost_optimization import summarise  # noqa: E402
from experiments.reporting import DATASET_LABELS, style_axes, write_run_log  # noqa: E402
from profiling.store import ProfileStore  # noqa: E402
from rules.llm_generator import BudgetExceeded, LLMClient, build_prompt  # noqa: E402
from rules.validator import accepted_rules, executable_rules, validate  # noqa: E402

REF_SCENARIO = "A_stable"      # T1..T5 are identical in every scenario
CONFIGS = ["Static rules", "LLM rules", "LLM + validation", "Static + validated LLM"]
CONFIG_COLORS = {"Static rules": "#eb6834", "LLM rules": "#e87ba4", "LLM + validation": "#2a78d6",
                 "Static + validated LLM": "#1baf7a"}
TOKENS_PER_RULE = 110          # used only for the expected-output estimate in --dry-run
REASONING_ALLOWANCE = 4000     # expected thinking/reasoning tokens per call (billed as output)


def llm_rules_dir(mock: bool) -> Path:
    return results_dir("llm_rules_mock" if mock else "llm_rules")


def prompts(datasets, exp):
    """Prompt per dataset. Example values come from the warm-up batches T1..T5 only (leakage fix,
    manuscript Section 5.2); the proof of concept sampled them from the whole cleaned source, which
    overlaps the evaluation batches. NOTE: this changes the prompts, so the cached proof-of-concept
    answers no longer match and a full rerun of this experiment makes new (paid) Bedrock calls.
    The confirmatory version is experiments/v2/llm_generate.py (development-pool examples)."""
    out = {}
    for i, (ds, cfg) in enumerate(datasets.items()):
        profile = ProfileStore(ds, REF_SCENARIO).load(batch_label(1))
        warm = pd.concat([load_batch(ds, REF_SCENARIO, t)[1] for t in range(1, exp["warmup_batches"] + 1)],
                         ignore_index=True)
        out[ds] = build_prompt(ds, cfg, profile, warm, exp["llm"]["sample_values"], seed=exp["seed"] + i)
    return out


def estimate(client: LLMClient, datasets, exp, prm) -> pd.DataFrame:
    rows = []
    llm = exp["llm"]
    for ds, cfg in datasets.items():
        n_in, method = client.count_input_tokens(prm[ds])
        n_cols = len(cfg["numeric"]) + len(cfg["categorical"])
        exp_out = min(llm["max_tokens"], (4 * n_cols + 2) * TOKENS_PER_RULE + REASONING_ALLOWANCE)
        cached = 0
        for r in range(1, llm["runs"] + 1):
            p = client.cache_root / ds / f"run_{r}.json"
            if p.exists():
                from rules.llm_generator import SYSTEM_PROMPT, prompt_hash
                h = prompt_hash(llm["model_id"], SYSTEM_PROMPT, prm[ds], llm["max_tokens"], llm.get("temperature"))
                cached += int(json.loads(p.read_text(encoding="utf-8")).get("prompt_hash") == h)
        new_calls = llm["runs"] - cached
        rows.append({"dataset": ds, "input_tokens_per_call": n_in, "token_count_method": method,
                     "expected_output_tokens_per_call": exp_out, "max_output_tokens_per_call": llm["max_tokens"],
                     "calls_needed": new_calls, "calls_cached": cached,
                     "expected_cost_usd": new_calls * client.cost(n_in, exp_out),
                     "worst_case_cost_usd": new_calls * client.cost(n_in, llm["max_tokens"])})
    return pd.DataFrame(rows)


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens and cost only; no LLM call")
    ap.add_argument("--mock", action="store_true", help="offline stand-in for the LLM (pipeline test only)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    llm = exp["llm"]
    client = LLMClient(llm, mock=True if args.mock else None)
    prm = prompts(datasets, exp)
    sfx = "_mock" if client.mock else ""          # mock outputs never overwrite real results

    est = estimate(client, datasets, exp, prm)
    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    print(f"Model: {llm['model_id']}   runs per dataset: {llm['runs']}   "
          f"prices: ${client.cfg['price_input_per_mtok']}/M input, ${client.cfg['price_output_per_mtok']}/M output"
          + ("   [MOCK - no Bedrock calls]" if client.mock else ""))
    print(est.round(4).to_string(index=False))
    print(f"TOTAL new calls: {int(est.calls_needed.sum())}   expected ${est.expected_cost_usd.sum():.2f}   "
          f"worst case ${est.worst_case_cost_usd.sum():.2f}   budget cap ${llm['max_cost_usd']:.2f}")
    if args.dry_run:
        est.to_csv(tables / f"exp3_cost_estimate{sfx}.csv", index=False)
        pdir = results_dir("llm_prompts"); pdir.mkdir(parents=True, exist_ok=True)
        for ds, p in prm.items():
            (pdir / f"{ds}.txt").write_text(p, encoding="utf-8")
        print(f"\nDry run only. Prompts written to {pdir} for inspection.")
        return

    # ---- 1. generate (or reuse cached answers)
    responses = {}
    try:
        for ds in datasets:
            n_in = int(est.loc[est.dataset == ds, "input_tokens_per_call"].iloc[0])
            for r in range(1, llm["runs"] + 1):
                rec = client.generate(ds, r, prm[ds], n_in)
                responses[(ds, r)] = rec
                src = "cache" if rec["from_cache"] else f"new call ${rec['cost_usd']:.4f}"
                print(f"[{ds}] run {r}: {rec['input_tokens']} in / {rec['output_tokens']} out tokens ({src}); "
                      f"{len(rec['rules'] or [])} rules; stop={rec['stop_reason']}"
                      + (f"; PARSE ERROR {rec['parse_error']}" if rec["parse_error"] else ""))
    except BudgetExceeded as exc:
        print(f"\nSTOPPED: {exc}. Raise llm.max_cost_usd in configs/experiment.yaml to continue.")
        if not responses:
            return
    print(f"New spend this run: ${client.spent_usd:.4f} over {client.calls} call(s)")

    # ---- 2. validate (Section 9)
    rule_rows, count_rows, rule_sets = [], [], {}
    for (ds, r), rec in responses.items():
        cfg = datasets[ds]
        warm = [load_batch(ds, REF_SCENARIO, t)[1] for t in range(1, exp["warmup_batches"] + 1)]
        cands = rec["rules"] or []
        val = validate(cands, cfg, warm, exp["validation"]["max_warmup_failure_rate"])
        acc = accepted_rules(val)
        raw = executable_rules(val, cands, cfg, list(warm[0].columns))
        rule_sets[(ds, r)] = {"validated": acc, "raw": raw}
        out_dir = llm_rules_dir(client.mock) / ds
        write_json({"model_id": rec["model_id"], "run": r, "mock": client.mock, "validation": val,
                    "validated_rules": acc, "raw_executable_rules": raw}, out_dir / f"run_{r}.json")
        for v in val:
            c = v["candidate"] if isinstance(v["candidate"], dict) else {}
            rule_rows.append({"dataset": ds, "run": r, "column": v["column"], "rule_type": v["rule_type"],
                              "status": v["status"], "stage": v["stage"], "detail": v["detail"],
                              "confidence": v["confidence"], "warmup_failure_rate": v["warmup_failure_rate"],
                              "condition": c.get("condition"), "reason": c.get("reason"),
                              "min": c.get("min"), "max": c.get("max"), "pattern": c.get("pattern"),
                              "allowed_values": json.dumps(c.get("allowed_values")) if c.get("allowed_values") else None})
        st = pd.Series([v["status"] for v in val]).value_counts()
        n_mon = len(cfg["numeric"]) + len(cfg["categorical"])
        cov_cols = {v["rule"]["column"] for v in val if v["status"] in ("accepted", "modified")
                    and v["rule"]["type"] not in ("completeness", "uniqueness")}
        count_rows.append({"dataset": ds, "run": r, "generated": len(val),
                           **{s: int(st.get(s, 0)) for s in ("accepted", "modified", "redundant", "rejected", "invalid")},
                           "valid": int(len(val) - st.get("invalid", 0)),
                           "columns_with_value_rule": len(cov_cols), "monitored_columns": n_mon,
                           "column_coverage": len(cov_cols) / n_mon,
                           "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"],
                           "cost_usd": rec["cost_usd"], "stop_reason": rec["stop_reason"],
                           "parse_error": rec["parse_error"]})
    rules_df, counts = pd.DataFrame(rule_rows), pd.DataFrame(count_rows)
    rules_df.to_csv(tables / f"exp3_rules{sfx}.csv", index=False)
    counts.to_csv(tables / f"exp3_rule_counts{sfx}.csv", index=False)

    # stability: overlap of (column, rule type) of validated rules between runs
    stab = []
    for ds in datasets:
        sets = {r: {(x.get("column") or "_records", x["type"]) for x in rule_sets[(ds, r)]["validated"]}
                for (d, r) in rule_sets if d == ds}
        pairs = [jaccard(sets[a], sets[b]) for a, b in combinations(sorted(sets), 2)]
        stab.append({"dataset": ds, "runs": len(sets), "mean_jaccard_validated_rules": float(np.mean(pairs)) if pairs else None})
    pd.DataFrame(stab).to_csv(tables / f"exp3_stability{sfx}.csv", index=False)

    # ---- 3. detection on the experimental batches
    rows = []
    for ds, cfg in datasets.items():
        for sc in exp["scenarios"]:
            rows += [dict(x, configuration="Static rules", run=0) for x in run(ds, sc, cfg, exp, "static")]
            for (d, r), rs in rule_sets.items():
                if d != ds:
                    continue
                rows += [dict(x, configuration="LLM rules", run=r) for x in run(ds, sc, cfg, exp, "fixed", extra_rules=rs["raw"])]
                rows += [dict(x, configuration="LLM + validation", run=r)
                         for x in run(ds, sc, cfg, exp, "fixed", extra_rules=rs["validated"])]
                rows += [dict(x, configuration="Static + validated LLM", run=r)
                         for x in run(ds, sc, cfg, exp, "static", extra_rules=rs["validated"])]
    batch = pd.DataFrame(rows)
    raw_path = results_dir("raw", f"exp3_batch_results{sfx}.csv"); raw_path.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw_path, index=False)
    ev = batch[batch.evaluation == 1]

    per_run = summarise(ev, ["dataset", "configuration", "run"])
    for t in ("missing", "duplicate", "invalid_category", "outlier"):
        g = ev.groupby(["dataset", "configuration", "run"])[[f"tp_{t}", f"n_{t}"]].sum().reset_index()
        per_run = per_run.merge(g.assign(**{f"recall_{t}": g[f"tp_{t}"] / g[f"n_{t}"]})[
            ["dataset", "configuration", "run", f"recall_{t}"]], on=["dataset", "configuration", "run"])
    metrics = ["precision", "recall", "f1", "fpr", "false_alarms", "defects_detected", "missed_defects",
               "recall_missing", "recall_duplicate", "recall_invalid_category", "recall_outlier"]
    agg = per_run.groupby(["dataset", "configuration"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    order = {c: i for i, c in enumerate(CONFIGS)}
    agg = agg.sort_values(["dataset", "configuration"], key=lambda s: s.map(order) if s.name == "configuration" else s)
    agg.to_csv(tables / f"exp3_detection{sfx}.csv", index=False)

    # additional coverage of validated LLM rules over static rules
    static_tp = per_run[per_run.configuration == "Static rules"].set_index("dataset")
    add = per_run[per_run.configuration == "Static + validated LLM"].copy()
    add["extra_defects_detected"] = add.defects_detected - add.dataset.map(static_tp.defects_detected)
    add["extra_false_alarms"] = add.false_alarms - add.dataset.map(static_tp.false_alarms)
    add[["dataset", "run", "extra_defects_detected", "extra_false_alarms"]].to_csv(
        tables / f"exp3_additional_coverage{sfx}.csv", index=False)

    # figure
    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ds_list = list(datasets)
    x = np.arange(len(ds_list)); w = 0.8 / len(CONFIGS)
    for ax, (m, label) in zip(axes, [("f1", "F1-score"), ("fpr", "False-positive rate")]):
        for j, c in enumerate(CONFIGS):
            sub = agg[agg.configuration == c].set_index("dataset").reindex(ds_list)
            ax.bar(x + (j - 1.5) * w, sub[f"{m}_mean"], w * 0.92, yerr=sub[f"{m}_std"].fillna(0),
                   color=CONFIG_COLORS[c], label=c, error_kw={"elinewidth": 1, "ecolor": "#55554f"})
        ax.set_xticks(x, [DATASET_LABELS[d].replace(" Inspections", "") for d in ds_list], fontsize=8.5)
        ax.set_title(label, fontsize=10, loc="left", color="#1f1f1c")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.99, 0.01, f"{llm['model_id']}, {llm['runs']} runs per dataset (mean, error bar = sd). All scenarios, T6-T20."
             + (" MOCK LLM." if client.mock else ""), ha="right", fontsize=7.5, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / ("exp3_llm_rules_mock.png" if client.mock else "exp3_llm_rules.png"), dpi=300)
    plt.close(fig)

    total_cost = float(sum(rec["cost_usd"] for rec in responses.values()))
    summary = {"model_id": llm["model_id"], "mock": client.mock,
               "llm_total_cost_usd_all_cached_calls": total_cost, "new_spend_usd": client.spent_usd,
               "tokens_in": int(counts.input_tokens.sum()), "tokens_out": int(counts.output_tokens.sum()),
               "rule_counts": counts.drop(columns=["parse_error"]).to_dict(orient="records")}
    log = write_run_log(f"exp3_llm_rules{sfx}", datasets, exp, batch, summary,
                        {"batch_results": str(raw_path), "tables": str(tables), "figures": str(figs),
                         "llm_responses": str(client.cache_root), "llm_rules": str(llm_rules_dir(client.mock))},
                        time.perf_counter() - t0)
    print()
    print(counts.drop(columns=["parse_error"]).to_string(index=False))
    print()
    print(agg[["dataset", "configuration", "precision_mean", "recall_mean", "f1_mean", "fpr_mean"]].round(4).to_string(index=False))
    print(f"\nLLM cost of all responses used: ${total_cost:.4f} (new spend this run ${client.spent_usd:.4f})")
    print(f"Run log: {log}")


if __name__ == "__main__":
    main()
