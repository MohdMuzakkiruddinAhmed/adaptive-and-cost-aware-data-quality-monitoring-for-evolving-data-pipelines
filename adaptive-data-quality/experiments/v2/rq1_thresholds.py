"""RQ1 / H1a: adaptive thresholds versus strong baselines (supervisor experiment 1).

Question   Does adaptation handle legitimate drift better than fixed and history-based baselines without
           losing error detection?
Methods    protocol.yaml baselines.methods: fixed_t1 (PoC baseline), fixed_pooled (all five warm-up
           batches, primary comparator), rolling mean/sd, rolling median/MAD, rolling IQR, EWMA, the proposed
           adaptive method (current batch) and adaptive_prev (previous batches only). AVH is batch-level and
           is evaluated in rq1_classification (native granularity).
           Established tool (protocol.yaml external_tools.rq1_methods): Amazon Deequ. Deequ suggests constraints
           from the warm-up (as observed, DEFAULT rules; or without its erroneous records, extended rules: the
           tool's best case) and verifies them on every evaluation batch itself; the records it reports as
           violating are its record-level verdict (baselines/external_tools.py deequ_flags). Skipped with a
           warning when the Deequ sidecar has not produced its outputs (tools/external_validators/README.md).
Scenarios  stable, drift-only, degradation-only, combined, zero-defect control, mild errors (stable/drift),
           sustained contamination (stable/drift): protocol.yaml experiments.rq1_thresholds.
Execution  every check on every batch (no selection), same base rules, same k = 3, same history window.
Measures   FPR, recall, precision, F1 per dataset and scenario; recall by error type, severity and column;
           FP per 10k valid records; paired seed-level CIs of (adaptive - comparator) for FPR and recall;
           recall non-inferiority against the prespecified margin (protocol.yaml criteria).

    python -m experiments.v2.rq1_thresholds --split dev --jobs 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import breakdown_long  # noqa: E402
from evaluation.stats import compare_methods  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, summarize_counts)
from monitor.loop import Monitor, MonitorConfig  # noqa: E402

EXPERIMENT = "rq1_thresholds"


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        for method in payload["methods"]:
            m = Monitor(cfg, exp, pr, MonitorConfig(name=method, method=method, classify=False), seed=seed)
            rows += run_monitor_over(rep, m, extra={"method": method})
        if payload.get("tool_methods"):
            rows += deequ_rows(rep, ds, seed, scen, cfg, exp, pr, payload)
    return pd.DataFrame(rows)


def deequ_rows(rep, ds: str, seed: int, scen: str, cfg, exp, pr, payload) -> list:
    """Evaluation rows of every Deequ method: the records Deequ itself reports as violating its suggested
    constraints (row-level verification results), optionally together with this project's base checks."""
    from baselines.external_tools import deequ_flags, load_verify
    from evaluation.metrics import record_counts
    from execution.local_executor import execute
    from monitor.suite import SuiteBuilder
    ver = load_verify(Path(payload["exchange"]), ds, seed, scen)
    base_suite = SuiteBuilder(cfg, exp, pr, "fixed_pooled", include_statistical=False, include_base=True)
    rows = []
    flags = {name: deequ_flags(ver, spec["variant"], spec["reading"]) for name, spec in payload["tool_methods"].items()}
    for t in range(rep.replay.warm + 1, rep.replay.n_batches + 1):
        b = rep.batch(t)
        ids = b.df["record_id"].to_numpy()
        base_failed = None
        valid_novel = set(b.novelty.loc[b.novelty.kind == "valid", "record_id"]) if len(b.novelty) else set()
        for name, spec in payload["tool_methods"].items():
            f = flags[name][t]
            if f["rows"] != len(b.df):
                raise RuntimeError(f"Deequ saw {f['rows']} rows in {ds}/seed {seed}/{scen}/T{t}, the replay has "
                                   f"{len(b.df)}: re-export the batches (run_all step export_tools)")
            failed = {ids[p] for p in f["positions"]}
            if spec.get("base"):
                if base_failed is None:
                    rules, _ = base_suite.build(list(b.df.columns), [], [], [], None)
                    base_failed = execute(rules, b.df)["failed_ids"]
                failed |= base_failed
            rows.append({**b.meta, "method": name, "scenario": scen, "seed": seed, "dataset": ds,
                         "split": rep.replay.split,
                         **record_counts(failed, b.gt, b.df["record_id"], valid_novel if valid_novel else None),
                         "deequ_constraints": f["constraints"], "deequ_failed_constraints": f["failed"],
                         "deequ_failed_list": f["failed_constraints"]})
    return rows


def deequ_report(ctx, payload) -> pd.DataFrame:
    """What Deequ suggested per warm-up and variant, by constraint type."""
    import re
    from baselines.external_tools import load_suggestions
    from experiments.v2.tool_exchange import exchange_dir, load_manifest
    man = load_manifest(ctx.split)
    variants = man.get("tools", {}).get("deequ", {}).get("suggestion_variants", {})
    rows, seen = [], set()
    for u in man["units"]:
        if not u.get("suggest") or u["dataset"] not in ctx.datasets:
            continue
        for v, spec in variants.items():
            h = u["warm_clean_hash"] if spec.get("warmup") == "clean" else u["warm_hash"]
            if (v, h) in seen:
                continue
            seen.add((v, h))
            sug = load_suggestions(exchange_dir(ctx.split), h, v)
            if sug is None:
                continue
            codes = sug.get("constraints_verified") or [x.get("code_for_constraint", "") for x in sug["suggestions"]]
            kinds = pd.Series([(re.match(r"\.(\w+)\(", c) or [None, "other"])[1] for c in codes]).value_counts()
            rows.append({"dataset": u["dataset"], "variant": v, "warm_hash": h, "warmup_rows": sug.get("rows"),
                         "constraints": len(codes), **{f"n_{k}": int(n) for k, n in kinds.items()},
                         "examples": " | ".join(codes[:6])})
    return pd.DataFrame(rows)


def tool_methods(ctx):
    """Deequ methods whose verification outputs exist for every required run; warns and drops them otherwise."""
    from experiments.v2.tool_exchange import exchange_dir, missing_outputs, tool_outputs_digest
    names = dict(ctx.protocol.get("external_tools", {}).get("rq1_methods", {}) or {})
    if not names:
        return {}, {}
    miss = missing_outputs(ctx.split, "deequ_verify", list(ctx.datasets), ctx.seeds, ctx.scenarios)
    if miss is None or miss > 0:
        why = "batches not exported" if miss is None else f"{miss} run(s) without current Deequ verification output"
        print(f"[{EXPERIMENT}] WARNING: Deequ methods skipped ({why}; see tools/external_validators/README.md)",
              flush=True)
        return {}, {}
    return names, {"exchange": str(exchange_dir(ctx.split)), "tool_outputs": tool_outputs_digest(ctx.split, ("deequ",))}


def main(argv=None) -> None:
    args = parser(__doc__).parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    base_methods = list(ctx.protocol["baselines"]["methods"])
    tmethods, tpayload = tool_methods(ctx)
    raw = run_units(ctx, worker, {"methods": base_methods, "tool_methods": tmethods, **tpayload})
    methods = base_methods + list(tmethods)
    ev = raw[raw.evaluation == 1]
    if tmethods:
        deequ_report(ctx, tpayload).to_csv(ctx.out / "rq1a_deequ_constraints.csv", index=False)
        # Deequ's batch-level verdict: share of evaluation batches with at least one failed suggested constraint,
        # on batches with and without degradation (one row per variant: the "failed" reading of each variant)
        dq = ev[ev.method.isin([m for m, sp in tmethods.items() if sp["reading"] == "failed"])].copy()
        dq["alarm"] = dq["deequ_failed_constraints"] > 0
        dq.groupby(["dataset", "scenario", "method", "degradation_label"]).agg(
            batches=("alarm", "size"), alarm_share=("alarm", "mean"),
            mean_failed_constraints=("deequ_failed_constraints", "mean"),
            constraints=("deequ_constraints", "mean")).reset_index() \
            .to_csv(ctx.out / "rq1a_deequ_batch_alarms.csv", index=False)

    table = summarize_counts(ev, ["dataset", "scenario", "method"])
    table.to_csv(ctx.out / "rq1a_record_level.csv", index=False)
    breakdown_long(ev, ["dataset", "scenario", "method"]).to_csv(ctx.out / "rq1a_breakdown.csv", index=False)

    per_seed = ev.groupby(["dataset", "scenario", "method", "seed"])[["TP", "FP", "FN", "TN", "P", "N"]].sum().reset_index()
    crit = ctx.criteria
    cis = []
    for comparator in methods:
        if comparator == "adaptive":
            continue
        sub = per_seed[per_seed.method.isin(["adaptive", comparator])]
        c = compare_methods(sub, ["dataset", "scenario"], "method", comparator, "seed",
                            {"recall": ("TP", "P"), "fpr": ("FP", "N")}, B=crit["bootstrap_replicates"],
                            level=crit["confidence_level"], margin=crit["recall_noninferiority_margin"])
        cis.append(c)
    ci = pd.concat(cis, ignore_index=True) if cis else pd.DataFrame()
    ci.to_csv(ctx.out / "rq1a_paired_ci.csv", index=False)

    # H1a decision table: lower FPR under drift AND recall non-inferior, per dataset (primary comparator)
    prim = ctx.protocol["baselines"]["primary_comparator"]
    if len(ci):
        dec = []
        for (ds, sc), g in ci[ci.comparator == prim].groupby(["dataset", "scenario"]):
            f = g[g.metric == "fpr"].iloc[0]
            r = g[g.metric == "recall"].iloc[0]
            # zero-defect control: no erroneous record, so recall is undefined and H1a reduces to the FPR part
            recall_applicable = bool(pd.notna(r.estimate))
            noninf = bool(r.noninferior) if recall_applicable else None
            dec.append({"dataset": ds, "scenario": sc, "comparator": prim,
                        "drift_scenario": bool(ctx.protocol["scenarios"][sc]["drift"] != "drift_none"),
                        "fpr_diff": f.estimate, "fpr_ci_high": f.ci_high, "lower_fpr": bool(f.ci_high < 0),
                        "recall_applicable": recall_applicable, "recall_diff": r.estimate,
                        "recall_ci_low": r.ci_low, "recall_noninferior": noninf,
                        "H1a_supported": bool(f.ci_high < 0 and (noninf if recall_applicable else True))})
        pd.DataFrame(dec).to_csv(ctx.out / "rq1a_h1a_decisions.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"methods": methods})
    print(table[["dataset", "scenario", "method", "precision", "recall", "f1", "fpr", "seeds"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
