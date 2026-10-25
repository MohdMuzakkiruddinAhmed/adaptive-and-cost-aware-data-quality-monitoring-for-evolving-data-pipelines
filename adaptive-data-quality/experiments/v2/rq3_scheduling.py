"""RQ3 / H3: risk-based scheduling with actual cost (supervisor experiment 5).

Schedulers  all (run every check) | periodic | random | failure_only | drift_only | risk (proposed), all on the
            SAME candidate suite (adaptive statistical rules + base checks; optional validated LLM rules) with
            the SAME mandatory checks (basic tier always runs). Periodic and random are cost-matched to the risk
            policy's executed share of standard checks in the same run (protocol.yaml schedulers.cost_match).
Incidents   sparse failures in one check family (H1 duplicates, H2 one category column), an incident shorter
            than the refresh interval (H3), incident + recovery (H4), sustained degradation (C, D), plus
            stable/drift controls (A, B).
Loop        classification and scheduling run in the same loop (monitor/loop.py); unexecuted checks give no
            evidence (per-check degradation rule); an evaluation-only full-suite reference pass measures what
            selection missed and never feeds the scheduler.
Measures    detection: recall, post-onset recall, incident detection delay, never-detected incidents, recovery
            time, missed-by-selection; cost: executed checks, logical bytes touched (secondary), and with
            --runtime: complete monitoring time by phase (read, profile, drift, adapt, select, execute, classify,
            persist), peak memory (separate tracemalloc pass), with monitor-computed and upstream profiles, with
            and without profile-statistics reuse (baselines reuse already computed statistics).

    python -m experiments.v2.rq3_scheduling --split dev --jobs 4          # detection + check counts
    python -m experiments.v2.rq3_scheduling --split dev --runtime         # timed runtime trials (single process!)
"""
from __future__ import annotations

import random
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import incident_summary, onset_delay  # noqa: E402
from evaluation.stats import compare_methods  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, summarize_counts)
from experiments.v2.llm_rules import load_generations, models  # noqa: E402
from monitor.loop import PHASES, Monitor, MonitorConfig  # noqa: E402
from rules.validator import accepted_rules, validate  # noqa: E402

EXPERIMENT = "rq3_scheduling"


def _llm_extra(src, ds, cfg, exp, warm_dfs, payload):
    if not payload.get("with_llm"):
        return []
    mk, mid = next(iter(payload["models"].items()))
    gens = load_generations(src, ds, mid, 1, payload.get("mock", False))
    if not gens:
        return []
    rec = gens[min(gens)]
    val = validate(rec["rules"] or [], cfg, warm_dfs, exp["validation"]["max_warmup_failure_rate"], "logical")
    return [dict(r, source="llm") for r in accepted_rules(val, prefix=f"{mk}1_")]


def monitor_config(name, policy, sch, extra, kw, reuse=True, persist=False):
    return MonitorConfig(name=name, method="adaptive", extra_rules=extra, scheduler=policy, scheduler_kw=kw,
                         drift_source=sch["drift_source"], degradation_rule=sch["degradation_rule"],
                         reuse_profile_stats=reuse, persist=persist)


def cost_matched_kw(policy: str, share: float, sch, exp):
    if policy == "periodic":
        if sch["cost_match"] and share > 0:
            return {"rate": share, "phase": sch["periodic_phase"]}
        return {"period": exp["risk"]["refresh_every"], "phase": sch["periodic_phase"]}
    if policy == "random":
        return {"rate": share if sch["cost_match"] else 1 / exp["risk"]["refresh_every"]}
    return {}


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    sch = pr["schedulers"]
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        extra = _llm_extra(src, ds, cfg, exp, [rep.batch(t).df for t in range(1, exp["warmup_batches"] + 1)], payload)
        risk_rows = run_monitor_over(rep, Monitor(cfg, exp, pr, monitor_config("risk", "risk", sch, extra, {}), seed=seed),
                                     reference=sch["reference_pass"], extra={"policy": "risk"})
        ev = [r for r in risk_rows if r["evaluation"] == 1]
        tot = sum(r.get("sched_standard_total") or 0 for r in ev)
        share = sum(r.get("sched_standard_executed") or 0 for r in ev) / tot if tot else 1.0
        for r in risk_rows:
            r["cost_matched_share"] = share
        rows += risk_rows
        for pol in sch["policies"]:
            if pol == "risk":
                continue
            kw = cost_matched_kw(pol, share, sch, exp)
            m = Monitor(cfg, exp, pr, monitor_config(pol, pol, sch, extra, kw), seed=seed)
            rows += run_monitor_over(rep, m, reference=sch["reference_pass"],
                                     extra={"policy": pol, "cost_matched_share": share, **{f"kw_{k}": v for k, v in kw.items()}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- incident-level summaries
def incident_table(raw: pd.DataFrame, protocol, crit) -> pd.DataFrame:
    out = []
    scen_cfg = protocol["scenarios"]
    for (ds, seed, scen, pol), g in raw.groupby(["dataset", "seed", "scenario", "policy"]):
        g = g.sort_values("t")
        ge = g[g.evaluation == 1].assign(label=lambda d: d.get("pred_label", pd.Series("", index=d.index)).astype(str))
        incs = scen_cfg[scen].get("incidents", [])
        for inc in incs:
            s = incident_summary(ge, inc["name"], inc["start"], inc["end"], crit["incident_detect_recall"], int(g.t.max()))
            out.append({"dataset": ds, "seed": seed, "scenario": scen, "policy": pol, **s})
        if not incs and (g.degradation_label == 1).any():     # sustained degradation (C, D)
            on = int(g.loc[g.degradation_label == 1, "t"].min())
            post = ge[ge.t >= on]
            rec_b = post.TP / post.P.where(post.P > 0)
            det = post.t[rec_b >= crit["incident_detect_recall"]]
            out.append({"dataset": ds, "seed": seed, "scenario": scen, "policy": pol, "incident": "degradation",
                        "start": on, "end": int(g.t.max()),
                        "post_onset_recall": post.TP.sum() / post.P.sum() if post.P.sum() else np.nan,
                        "detection_delay": float(det.iloc[0] - on) if len(det) else np.nan, "detected": bool(len(det)),
                        "label_delay": onset_delay(ge.t, ge.label.str.contains("degradation"), on),
                        "recovery_time": np.nan, "false_alarms_after_recovery": 0})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- runtime trials
def runtime_trials(ctx: Context, shares: pd.DataFrame) -> pd.DataFrame:
    src = data_source()
    pr, exp = ctx.protocol, ctx.exp
    sch, rt = pr["schedulers"], pr["cost"]["runtime_trials"]
    seed = ctx.seeds[0]
    scen_list = [s for s in ("A_stable", "D_drift_degradation", "H1_sparse_duplicates") if s in ctx.scenarios] or ctx.scenarios[:1]
    rows = []
    for ds, cfg in ctx.datasets.items():
        for scen in scen_list:
            rep = src.replay(ds, ctx.split, seed, scen, cache=False)
            with tempfile.TemporaryDirectory(prefix="adq_rt_") as tmp:
                paths = []
                for t in range(1, exp["n_batches"] + 1):
                    p = Path(tmp) / f"T{t:02d}.parquet"
                    rep.batch(t).df.to_parquet(p, index=False)
                    paths.append(p)
                sel = shares[(shares.dataset == ds) & (shares.scenario == scen)]
                share = float(sel.cost_matched_share.iloc[0]) if len(sel) else 0.5
                configs = [(pol, reuse) for pol in sch["policies"] for reuse in (False, True)]
                for r in range(rt["repeats"]):
                    order = list(configs)
                    if rt.get("randomize_order", True):
                        random.Random(1000 * r + seed).shuffle(order)
                    for pol, reuse in order:
                        mc = monitor_config(pol, pol, sch, [], cost_matched_kw(pol, share, sch, exp), reuse=reuse, persist=True)
                        m = Monitor(cfg, exp, pr, mc, seed=seed)
                        for t, p in enumerate(paths, 1):
                            st = m.step(p, f"T{t:02d}")
                            rows.append({"dataset": ds, "scenario": scen, "seed": seed, "policy": pol, "reuse": reuse,
                                         "repeat": r, "t": t, "warmup": st["warmup"], "rules_executed": st["rules_executed"],
                                         **{f"ms_{k}": v for k, v in st["ms"].items()}, "ms_total": st["ms_total"],
                                         "ms_total_upstream_profile": st["ms_total_upstream_profile"],
                                         "calibrate_ms": st["calibrate_ms"]})
                        m.close()
                if pr["cost"].get("measure_memory", True):          # separate, untimed pass
                    for pol, reuse in configs:
                        mc = monitor_config(pol, pol, sch, [], cost_matched_kw(pol, share, sch, exp), reuse=reuse, persist=True)
                        m = Monitor(cfg, exp, pr, mc, seed=seed)
                        for t, p in enumerate(paths, 1):
                            tracemalloc.start()
                            m.step(p, f"T{t:02d}")
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                            rows.append({"dataset": ds, "scenario": scen, "seed": seed, "policy": pol, "reuse": reuse,
                                         "repeat": -1, "t": t, "peak_memory_mb": peak / 1e6})
                        m.close()
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--runtime", action="store_true", help="timed runtime trials (run alone, single process)")
    ap.add_argument("--with-llm", action="store_true", help="add validated LLM rules (primary model, generation 1)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    crit = ctx.criteria
    payload = {"with_llm": args.with_llm, "models": models(ctx.protocol), "mock": args.mock}
    sfx = "_llm" if args.with_llm else ""
    raw = run_units(ctx, worker, payload, suffix=sfx)
    ev = raw[raw.evaluation == 1]

    table = summarize_counts(ev, ["dataset", "scenario", "policy"])
    extra = ev.groupby(["dataset", "scenario", "policy"]).agg(
        standard_executed=("sched_standard_executed", "sum"), standard_total=("sched_standard_total", "sum"),
        mean_coverage=("sched_coverage", "mean"), max_staleness=("sched_max_staleness", "max"),
        missed_by_selection=("missed_by_selection", "sum") if "missed_by_selection" in ev else ("TP", "size"),
        cost_matched_share=("cost_matched_share", "mean")).reset_index()
    table = table.merge(extra, on=["dataset", "scenario", "policy"])
    allr = table[table.policy == "all"].set_index(["dataset", "scenario"])
    idx = table.set_index(["dataset", "scenario"]).index
    table["saving_checks_pct"] = 100 * (1 - table.rules_executed.to_numpy() / allr.rules_executed.reindex(idx).to_numpy())
    table["saving_bytes_pct"] = 100 * (1 - table.bytes_scanned.to_numpy() / allr.bytes_scanned.reindex(idx).to_numpy())
    table.to_csv(ctx.out / f"rq3_detection_cost{sfx}.csv", index=False)

    inc = incident_table(raw, ctx.protocol, crit)
    inc.to_csv(ctx.out / f"rq3_incidents_per_run{sfx}.csv", index=False)
    if len(inc):
        inc.groupby(["dataset", "scenario", "incident", "policy"]).agg(
            runs=("seed", "nunique"), post_onset_recall=("post_onset_recall", "mean"),
            mean_detection_delay=("detection_delay", "mean"), never_detected=("detected", lambda s: int(s.eq(False).sum())),
            mean_label_delay=("label_delay", "mean"), mean_recovery_time=("recovery_time", "mean")).reset_index() \
            .to_csv(ctx.out / f"rq3_incidents{sfx}.csv", index=False)

    per_seed = ev.groupby(["dataset", "scenario", "policy", "seed"])[["TP", "P", "FP", "N"]].sum().reset_index()
    cis = [compare_methods(per_seed, ["dataset", "scenario"], "policy", ref, "seed",
                           {"recall": ("TP", "P"), "fpr": ("FP", "N")}, B=crit["bootstrap_replicates"],
                           level=crit["confidence_level"]).assign(reference=ref) for ref in ("all", "risk")]
    pd.concat(cis, ignore_index=True).to_csv(ctx.out / f"rq3_paired_ci{sfx}.csv", index=False)

    # H3 "useful": fewer executed checks than run-all with post-onset recall >= target (degradation scenarios)
    if len(inc):
        po = inc.groupby(["dataset", "scenario", "policy"])["post_onset_recall"].mean().reset_index()
        dec = table.merge(po, on=["dataset", "scenario", "policy"], how="left")
        dec["meets_detection_target"] = dec.post_onset_recall >= crit["detection_target_recall"]
        dec["useful"] = dec.meets_detection_target & (dec.saving_checks_pct > 0)
        dec[["dataset", "scenario", "policy", "saving_checks_pct", "saving_bytes_pct", "post_onset_recall",
             "meets_detection_target", "useful"]].to_csv(ctx.out / f"rq3_h3_decisions{sfx}.csv", index=False)

    if args.runtime:
        shares = raw.groupby(["dataset", "scenario"])["cost_matched_share"].first().reset_index()
        rt = runtime_trials(ctx, shares)
        rt.to_csv(ctx.out / f"rq3_runtime_trials_raw{sfx}.csv", index=False)
        timed = rt[(rt.repeat >= 0) & (~rt.warmup.astype(bool))]
        cols = [f"ms_{p}" for p in PHASES] + ["ms_total", "ms_total_upstream_profile"]
        per_rep = timed.groupby(["dataset", "scenario", "policy", "reuse", "repeat"])[cols].sum().reset_index()
        rsum = per_rep.groupby(["dataset", "scenario", "policy", "reuse"])[cols].agg(["median", lambda s: s.quantile(.75) - s.quantile(.25)])
        rsum.columns = [f"{c}_{'median' if s == 'median' else 'iqr'}" for c, s in rsum.columns]
        mem = rt[rt.repeat == -1].groupby(["dataset", "scenario", "policy", "reuse"])["peak_memory_mb"].max().rename("peak_memory_mb")
        rsum.join(mem).reset_index().to_csv(ctx.out / f"rq3_runtime{sfx}.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"with_llm": args.with_llm, "runtime_trials": args.runtime})
    print(table[["dataset", "scenario", "policy", "recall", "precision", "rules_executed", "saving_checks_pct"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
