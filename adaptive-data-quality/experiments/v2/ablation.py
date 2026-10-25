"""Complete-system and component ablation (supervisor experiment 6).

Factors     adaptation      fixed_pooled | adaptive
            LLM rules       none | validated (primary model, logical redundancy, every generation; mean over
                            generations, seeds remain the replication unit)
            selection       all (every check) | risk (proposed scheduler)
            = 8 combinations, including the previously missing "adaptive + validated LLM + every check",
            which separates the effect of semantic rules from the effect of scheduling. Reference row:
            fixed_t1 rules, run-all (proof-of-concept static baseline).
Loop        classification and scheduling in the same loop for every combination: gated drift built from the
            executed checks, per-check degradation rule (robust to skipped checks), coverage and staleness
            recorded; reference full-suite pass for evaluation only.
Measures    record-level detection (precision, recall, F1, FPR, semantic recall), batch labels (macro-F1,
            balanced accuracy), executed checks and monitoring time (profiles supplied upstream; full timing in
            rq3 --runtime), detection delays; paired seed-level CIs against "adaptive + none + all" and
            factor main effects.

    python -m experiments.v2.ablation --split dev --jobs 4
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import label_metrics, onset_delay  # noqa: E402
from evaluation.stats import compare_methods, paired_mean_diff  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, summarize_counts, true_class)
from experiments.v2.llm_rules import load_generations, models  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402
from rules.validator import accepted_rules, validate  # noqa: E402

EXPERIMENT = "ablation"
REFERENCE = "adaptive|none|all"


def combos(ab):
    out = [(a, l, s) for a, l, s in itertools.product(ab["adaptation"], ab["llm_rules"], ab["selection"])]
    out += [(r, "none", "all") for r in ab.get("reference_rows", [])]
    return out


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    sch = pr["schedulers"]
    mk, mid = next(iter(payload["models"].items()))
    gens = load_generations(src, ds, mid, payload["generations"], payload["mock"])
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        warm = [rep.batch(t).df for t in range(1, exp["warmup_batches"] + 1)]
        llm_sets = {}
        for g, rec in gens.items():
            val = validate(rec["rules"] or [], cfg, warm, exp["validation"]["max_warmup_failure_rate"], "logical")
            llm_sets[g] = [dict(r, source="llm") for r in accepted_rules(val, prefix=f"{mk}{g}_")]
        for a, l, s in combos(payload["ablation"]):
            variants = {0: []} if l == "none" else llm_sets
            if l != "none" and not variants:
                continue
            for g, extra in variants.items():
                mc = MonitorConfig(name=f"{a}|{l}|{s}", method=a, extra_rules=extra, scheduler=s,
                                   drift_source=sch["drift_source"], degradation_rule=sch["degradation_rule"],
                                   reuse_profile_stats=True)
                m = Monitor(cfg, exp, pr, mc, seed=seed)
                rows += run_monitor_over(rep, m, reference=(s != "all"),
                                         extra={"config": mc.name, "adaptation": a, "llm": l, "selection": s,
                                                "generation": g})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    pr, crit = ctx.protocol, ctx.criteria
    payload = {"ablation": pr["ablation"], "models": models(pr, ["primary"]), "generations": pr["rq2"]["generations"],
               "mock": args.mock}
    sfx = "_mock" if args.mock else ""
    raw = run_units(ctx, worker, payload, suffix=sfx)
    ev = raw[raw.evaluation == 1].copy()
    ev["true_class"] = ev.apply(true_class, axis=1)
    if "n_type:semantic_numeric" in ev or "n_type:semantic_category" in ev:
        sn = [c for c in ev.columns if c.startswith("n_type:semantic")]
        ev["sem_n"] = ev[sn].fillna(0).sum(axis=1)
        ev["sem_tp"] = ev[[c.replace("n_", "tp_", 1) for c in sn]].fillna(0).sum(axis=1)
    else:
        ev["sem_n"] = ev["sem_tp"] = 0

    # detection: pool counts per generation, then average generations (LLM configs) within dataset
    keys = ["dataset", "config", "adaptation", "llm", "selection"]
    pg = summarize_counts(ev, keys + ["generation"])
    sem = ev.groupby(keys + ["generation"])[["sem_n", "sem_tp", "ms_total"]].sum().reset_index()
    pg = pg.drop(columns=["ms_total"], errors="ignore").merge(sem, on=keys + ["generation"])
    pg["semantic_recall"] = pg.sem_tp / pg.sem_n.where(pg.sem_n > 0)
    labs = []
    for k, g in ev.groupby(keys + ["generation"]):
        labs.append({**dict(zip(keys + ["generation"], k)), **{m: v for m, v in label_metrics(g.true_class, g.pred_label).items()
                                                              if m in ("macro_f1", "balanced_accuracy", "accuracy",
                                                                       "false_degradation_during_drift", "insufficient_share")}})
    pg = pg.merge(pd.DataFrame(labs), on=keys + ["generation"])
    metrics = ["precision", "recall", "f1", "fpr", "semantic_recall", "macro_f1", "balanced_accuracy",
               "false_degradation_during_drift", "rules_executed", "bytes_scanned", "ms_total"]
    table = pg.groupby(keys)[metrics].mean().reset_index()
    table.to_csv(ctx.out / f"ablation_summary{sfx}.csv", index=False)

    # paired CIs vs adaptive|none|all (counts pooled over generations within seed)
    ps = ev.groupby(["dataset", "config", "seed"])[["TP", "P", "FP", "N", "sem_tp", "sem_n"]].sum().reset_index()
    ci = compare_methods(ps, ["dataset"], "config", REFERENCE, "seed",
                         {"recall": ("TP", "P"), "fpr": ("FP", "N"), "semantic_recall": ("sem_tp", "sem_n")},
                         B=crit["bootstrap_replicates"], level=crit["confidence_level"])
    lab_seed = [{"dataset": d, "config": c, "seed": s, **label_metrics(g.true_class, g.pred_label)}
                for (d, c, s), g in ev.groupby(["dataset", "config", "seed"])]
    ls = pd.DataFrame(lab_seed)
    rows = []
    for d, g in ls.groupby("dataset"):
        ref = g[g.config == REFERENCE].set_index("seed")
        for c, gc in g.groupby("config"):
            if c == REFERENCE:
                continue
            gc = gc.set_index("seed")
            common = sorted(set(gc.index) & set(ref.index))
            r = paired_mean_diff(gc.loc[common, "macro_f1"], ref.loc[common, "macro_f1"], crit["bootstrap_replicates"])
            rows.append({"dataset": d, "method": c, "comparator": REFERENCE, "metric": "macro_f1", **r})
    ci = pd.concat([ci, pd.DataFrame(rows)], ignore_index=True)
    ci.to_csv(ctx.out / f"ablation_paired_ci{sfx}.csv", index=False)

    # factor main effects: mean difference between levels, holding the other factors' levels balanced
    eff = []
    core = table[table.adaptation.isin(pr["ablation"]["adaptation"])]
    for factor, (lo, hi) in {"adaptation": tuple(pr["ablation"]["adaptation"]), "llm": ("none", "validated"),
                             "selection": ("all", "risk")}.items():
        for d, g in core.groupby("dataset"):
            a, b = g[g[factor] == hi][metrics].mean(), g[g[factor] == lo][metrics].mean()
            eff.append({"dataset": d, "factor": factor, "effect": f"{hi} - {lo}", **(a - b).to_dict()})
    pd.DataFrame(eff).to_csv(ctx.out / f"ablation_main_effects{sfx}.csv", index=False)

    dl = []
    for (d, c, s, sc), g in raw.groupby(["dataset", "config", "seed", "scenario"]):
        on = int(g.loc[g.degradation_label == 1, "t"].min()) if (g.degradation_label == 1).any() else None
        ge = g[g.evaluation == 1]
        dl.append({"dataset": d, "config": c, "seed": s, "scenario": sc,
                   "degradation_label_delay": onset_delay(ge.t, ge.pred_degradation.fillna(False).astype(bool), on)})
    pd.DataFrame(dl).groupby(["dataset", "config", "scenario"])["degradation_label_delay"].mean().reset_index() \
        .to_csv(ctx.out / f"ablation_delays{sfx}.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"mock": args.mock})
    print(table[["dataset", "config", "precision", "recall", "f1", "semantic_recall", "macro_f1", "rules_executed"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
