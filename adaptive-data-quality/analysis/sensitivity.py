"""Section 14: sensitivity analysis of the frozen v2.2 protocol.

Purpose     Show that the conclusions do not depend on one manually chosen setting (project description,
            Section 14). The legacy script experiments/sensitivity.py measured the proof-of-concept pipeline;
            this one re-runs the FROZEN v2.2 monitor (the code of the confirmatory test run) on the same replays,
            seeds and splits as the main experiments.
Design      One setting at a time (analysis/sensitivity.yaml; values from configs/experiment.yaml "sensitivity"
            where they existed before freezing). Every other setting stays frozen. Ten test seeds per setting,
            so the seed-to-seed spread of the default setting is the yardstick for every change.

    Dial              Values                                   Evaluated on
    defect_rate       1 (default), 3, 5, 10, 20 % of records    H1a: A_stable, B_drift
    outlier_sigma     3-5, 4-12 (default), 5-8, 8-12 sd         H1a
    range_k           2.5, 3 (default), 3.5, 4                  H1a (added after freezing)
    history_window    3, 5 (default), 8, 10 batches             H1a (added after freezing)
    calibration_z     2, 2.5, 3 (default), 3.5, 4               H1b: A-D, failure-rate threshold mean + z sd
    drift_alpha       0.01, 0.025, 0.05 (default), 0.10, 0.20   H1b (added after freezing)
    risk_weights      each weight x0, x0.5, x2; equal weights   H3: C, D, H1-H4
    risk_thresholds   moderate, high, very high, one at a time  H3

Claims re-checked in every setting (sens_conclusions.csv), with the prespecified criteria of protocol.yaml:
    C1  H1a: adaptive FPR lower than fixed_pooled under drift (paired seed CI upper bound < 0)
    C2  H1a: adaptive recall non-inferior to fixed_pooled under drift (CI lower bound > -margin)
    C3  adaptive FPR not above the better calibrated rolling baseline (ewma, rolling_mad_cal) under drift
    C4  H1b: proposed macro-F1 >= gated_fixed_pooled macro-F1 (scenarios A-D)
    C5  H3: risk selection saves checks with post-onset recall >= detection target in sustained degradation (C, D)

Outputs     results/v2/sensitivity/<split>/: sens_long.csv (every metric, setting, dataset, scenario),
            sens_conclusions.csv, sens_conclusions_summary.csv, sens_seed_spread.csv, manifest.json, raw/ (resumable)

    python -m analysis.sensitivity --split test --jobs 4            # full analysis (about 1 to 2 h)
    python -m analysis.sensitivity --split test --summary-only      # rebuild tables from raw/
    python -m analysis.sensitivity --split dev --seeds 42 --datasets chicago_food --dials range_k   # quick check
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.provenance import file_hash  # noqa: E402
from evaluation.metrics import label_metrics  # noqa: E402
from evaluation.stats import compare_methods  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, summarize_counts, true_class)
from experiments.v2.rq3_scheduling import incident_table  # noqa: E402
from injection.scenario import DataSource  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402

EXPERIMENT = "sensitivity"
HERE = Path(__file__).resolve().parent
SETTINGS = HERE / "sensitivity.yaml"
DIALS = ["defect_rate", "outlier_sigma", "range_k", "history_window", "calibration_z", "drift_alpha",
         "risk_weights", "risk_thresholds"]
DEFAULT = "default"


# ----------------------------------------------------------------------------- settings
def load_settings() -> Dict[str, Any]:
    return yaml.safe_load(SETTINGS.read_text())


def weight_variants(w: Dict[str, float], factors) -> Dict[str, Dict[str, float]]:
    """Each risk weight x factor (renormalised to sum 1), plus equal weights (as in the legacy analysis)."""
    v = {"equal (0.2 each)": {k: 1 / len(w) for k in w}}
    for comp in w:
        for f in factors:
            raw = {k: (x * f if k == comp else x) for k, x in w.items()}
            tot = sum(raw.values())
            v[f"{comp} removed" if f == 0 else f"{comp} x{f:g}"] = {k: x / tot for k, x in raw.items()}
    return v


def dial_values(exp: Dict[str, Any], pr: Dict[str, Any], st: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Non-default settings of every dial: [{"setting": label, "value": printable, "mod": {...}}]."""
    sens, add = exp["sensitivity"], st["added_dials"]
    out: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIALS}
    base_rate = exp["schedules"]["defects_baseline"][0]
    for r in sens["defect_rates"]:
        if r != base_rate:
            out["defect_rate"].append({"setting": f"{r:.0%}", "value": r, "mod": {"defect_rate": r}})
    for lab, sig in sens["outlier_sigma"].items():
        if list(sig) != list(exp["defects"]["outlier_sigma"]):
            out["outlier_sigma"].append({"setting": lab, "value": f"{sig[0]}-{sig[1]}", "mod": {"outlier_sigma": list(sig)}})
    for k in add["range_k"]:
        if k != exp["range_k"]:
            out["range_k"].append({"setting": f"k = {k:g}", "value": k, "mod": {"range_k": k}})
    for w in add["history_window"]:
        if w != exp["adaptive"]["history_window"]:
            out["history_window"].append({"setting": f"window {w}", "value": w, "mod": {"history_window": w}})
    for z in sens["calibration_z"]:
        if z != exp["classification"]["calibration_z"]:
            out["calibration_z"].append({"setting": f"z = {z:g}", "value": z, "mod": {"calibration_z": z}})
    a0 = pr["classification"]["drift_threshold"]["alpha"]
    for a in add["drift_alpha"]:
        if a != a0:
            out["drift_alpha"].append({"setting": f"alpha = {a:g}", "value": a, "mod": {"drift_alpha": a}})
    for lab, w in weight_variants(exp["risk"]["weights"], sens["weight_factors"]).items():
        out["risk_weights"].append({"setting": lab, "value": " ".join(f"{k}={v:.2f}" for k, v in w.items()),
                                    "mod": {"risk_weights": w}})
    t0 = exp["risk"]["thresholds"]
    for level, values in sens["risk_thresholds"].items():
        for v in values:
            t = dict(t0, **{level: v})
            if v == t0[level] or not t["moderate"] < t["high"] < t["very_high"]:
                continue
            out["risk_thresholds"].append({"setting": f"{level} = {v:g}", "value": v, "mod": {"risk_thresholds": t}})
    return out


def apply_exp(exp: Dict[str, Any], mod: Dict[str, Any]) -> Dict[str, Any]:
    e = copy.deepcopy(exp)
    if "defect_rate" in mod:
        e["schedules"]["defects_baseline"] = [mod["defect_rate"]] * e["n_batches"]
    if "outlier_sigma" in mod:
        e["defects"]["outlier_sigma"] = list(mod["outlier_sigma"])
    if "range_k" in mod:
        e["range_k"] = mod["range_k"]
    if "history_window" in mod:
        e["adaptive"]["history_window"] = mod["history_window"]
    if "calibration_z" in mod:
        e["classification"]["calibration_z"] = mod["calibration_z"]
    return e


DATA_MODS = ("defect_rate", "outlier_sigma")


# ----------------------------------------------------------------------------- worker
def _h1a(rep, cfg, exp, pr, seed, methods, tag) -> List[Dict[str, Any]]:
    rows = []
    for m in methods:
        mon = Monitor(cfg, exp, pr, MonitorConfig(name=m, method=m, classify=False), seed=seed)
        rows += run_monitor_over(rep, mon, extra={**tag, "family": "h1a", "arm": m})
    return rows


def _h1b(rep, cfg, exp, pr, seed, comparators, tag, alpha=None) -> List[Dict[str, Any]]:
    rows = []
    comps = pr["classification"]["comparators"]
    dt = dict(pr["classification"]["drift_threshold"])
    if alpha is not None:
        dt["alpha"] = alpha
    for c in comparators:
        spec = comps[c]
        mc = MonitorConfig(name=c, method=spec["failure"], drift_source=spec["drift"], degradation_rule="union_rate",
                           scheduler="all", drift_threshold=dt)
        rows += run_monitor_over(rep, Monitor(cfg, exp, pr, mc, seed=seed), extra={**tag, "family": "h1b", "arm": c})
    return rows


def _h3(rep, cfg, exp, pr, seed, tag, kw=None, with_all=False) -> List[Dict[str, Any]]:
    sch = pr["schedulers"]
    rows = []
    for pol in (["all"] if with_all else []) + ["risk"]:
        mc = MonitorConfig(name=pol, method="adaptive", scheduler=pol, scheduler_kw=(kw or {}) if pol == "risk" else {},
                           drift_source=sch["drift_source"], degradation_rule=sch["degradation_rule"],
                           reuse_profile_stats=True)
        rows += run_monitor_over(rep, Monitor(cfg, exp, pr, mc, seed=seed), extra={**tag, "family": "h3", "arm": pol})
    return rows


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    """All settings of one (dataset, seed). Replays are built one scenario at a time to bound memory."""
    src = data_source()
    cfg, exp0, pr = src.datasets[ds], src.exp, src.protocol
    st, dials, split = payload["settings"], payload["dials"], payload["split"]
    values = dial_values(exp0, pr, st)
    h1a, h1b, h3 = st["h1a"], st["h1b"], st["h3"]
    use = lambda fam: any(d in dials for d in fam["dials"])  # noqa: E731
    rows: List[Dict[str, Any]] = []
    D = {"dial": DEFAULT, "setting": DEFAULT, "value": ""}

    # 1. settings that keep the data: default replay of each scenario, every method-only dial on it
    scen_all = list(dict.fromkeys((h1a["scenarios"] if use(h1a) else []) + (h1b["scenarios"] if use(h1b) else [])
                                  + (h3["scenarios"] if use(h3) else [])))
    for scen in scen_all:
        rep = ProfiledReplay(src.replay(ds, split, seed, scen))
        if use(h1a) and scen in h1a["scenarios"]:
            rows += _h1a(rep, cfg, exp0, pr, seed, h1a["methods"], D)
            for dial in ("range_k", "history_window"):
                if dial in dials:
                    for v in values[dial]:
                        rows += _h1a(rep, cfg, apply_exp(exp0, v["mod"]), pr, seed, h1a["methods"],
                                     {"dial": dial, "setting": v["setting"], "value": str(v["value"])})
        if use(h1b) and scen in h1b["scenarios"]:
            rows += _h1b(rep, cfg, exp0, pr, seed, h1b["comparators"], D)
            if "calibration_z" in dials:
                for v in values["calibration_z"]:
                    rows += _h1b(rep, cfg, apply_exp(exp0, v["mod"]), pr, seed, h1b["comparators"],
                                 {"dial": "calibration_z", "setting": v["setting"], "value": str(v["value"])})
            if "drift_alpha" in dials:
                for v in values["drift_alpha"]:
                    rows += _h1b(rep, cfg, exp0, pr, seed, h1b["comparators"],
                                 {"dial": "drift_alpha", "setting": v["setting"], "value": str(v["value"])},
                                 alpha=v["mod"]["drift_alpha"])
        if use(h3) and scen in h3["scenarios"]:
            rows += _h3(rep, cfg, exp0, pr, seed, D, with_all=True)
            for dial, key in (("risk_weights", "weights"), ("risk_thresholds", "thresholds")):
                if dial in dials:
                    for v in values[dial]:
                        rows += _h3(rep, cfg, exp0, pr, seed,
                                    {"dial": dial, "setting": v["setting"], "value": str(v["value"])},
                                    kw={key: v["mod"][dial]})
        del rep

    # 2. settings that change the data: new replays from a copy of the configuration (same split pools)
    for dial in DATA_MODS:
        if dial not in dials or not use(h1a):
            continue
        for v in values[dial]:
            e = apply_exp(exp0, v["mod"])
            s2 = DataSource(src.datasets, e, pr)
            s2._pools = src._pools                       # same records, only the injection changes
            for scen in h1a["scenarios"]:
                rep = ProfiledReplay(s2.replay(ds, split, seed, scen))
                rows += _h1a(rep, cfg, e, pr, seed, h1a["methods"],
                             {"dial": dial, "setting": v["setting"], "value": str(v["value"])})
                del rep
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summaries
def h1a_tables(ev: pd.DataFrame, crit) -> tuple:
    keys = ["dial", "setting", "value", "dataset", "scenario", "arm"]
    table = summarize_counts(ev, keys)
    per_seed = ev.groupby(keys + ["seed"])[["TP", "FP", "P", "N"]].sum().reset_index()
    ci = compare_methods(per_seed[per_seed.arm.isin(["adaptive", "fixed_pooled"])],
                         ["dial", "setting", "value", "dataset", "scenario"], "arm", "fixed_pooled", "seed",
                         {"recall": ("TP", "P"), "fpr": ("FP", "N")}, B=crit["bootstrap_replicates"],
                         level=crit["confidence_level"], margin=crit["recall_noninferiority_margin"])
    return table, ci, per_seed


def h1b_tables(ev: pd.DataFrame) -> tuple:
    ev = ev.copy()
    ev["true_class"] = ev.apply(true_class, axis=1)
    rows, seed_rows = [], []
    for k, g in ev.groupby(["dial", "setting", "value", "dataset", "arm"]):
        lm = label_metrics(g.true_class, g.pred_label)
        stable = g.true_class == "stable"
        drift_true = g.drift_label.astype(bool)
        deg_true = g.degradation_label.astype(bool)
        pl = g.pred_label.astype(str)
        rows.append({**dict(zip(["dial", "setting", "value", "dataset", "arm"], k)),
                     "macro_f1": lm["macro_f1"], "balanced_accuracy": lm["balanced_accuracy"],
                     "false_drift_stable": float(pl[stable].str.contains("drift").mean()) if stable.any() else np.nan,
                     "drift_recall": float(pl[drift_true].str.contains("drift").mean()) if drift_true.any() else np.nan,
                     "degradation_recall": float(pl[deg_true].str.contains("degradation").mean()) if deg_true.any() else np.nan,
                     "false_degradation_during_drift": lm["false_degradation_during_drift"], "batches": len(g)})
        for s, gs in g.groupby("seed"):
            seed_rows.append({**dict(zip(["dial", "setting", "value", "dataset", "arm"], k)), "seed": s,
                              "macro_f1": label_metrics(gs.true_class, gs.pred_label)["macro_f1"]})
    return pd.DataFrame(rows), pd.DataFrame(seed_rows)


def h3_tables(raw: pd.DataFrame, pr, crit) -> pd.DataFrame:
    out = []
    allr = raw[(raw.dial == DEFAULT) & (raw.arm == "all")]
    for (dial, setting, value), g in raw[raw.arm == "risk"].groupby(["dial", "setting", "value"]):
        both = pd.concat([g, allr], ignore_index=True).assign(policy=lambda d: d.arm)
        inc = incident_table(both, pr, crit)
        po = inc.groupby(["dataset", "scenario", "policy"])["post_onset_recall"].mean().reset_index() if len(inc) \
            else pd.DataFrame(columns=["dataset", "scenario", "policy", "post_onset_recall"])
        ev = both[both.evaluation == 1]
        cost = ev.groupby(["dataset", "scenario", "policy"])[["rules_executed", "TP", "P"]].sum().reset_index()
        cost = cost.merge(po, on=["dataset", "scenario", "policy"], how="left")
        a = cost[cost.policy == "all"].set_index(["dataset", "scenario"])
        r = cost[cost.policy == "risk"].set_index(["dataset", "scenario"])
        for idx, rr in r.iterrows():
            saving = 100 * (1 - rr.rules_executed / a.loc[idx, "rules_executed"])
            out.append({"dial": dial, "setting": setting, "value": value, "dataset": idx[0], "scenario": idx[1],
                        "saving_checks_pct": saving, "recall": rr.TP / rr.P if rr.P else np.nan,
                        "run_all_recall": a.loc[idx, "TP"] / a.loc[idx, "P"] if a.loc[idx, "P"] else np.nan,
                        "post_onset_recall": rr.post_onset_recall,
                        "meets_detection_target": bool(rr.post_onset_recall >= crit["detection_target_recall"]),
                        "useful": bool(rr.post_onset_recall >= crit["detection_target_recall"] and saving > 0)})
    return pd.DataFrame(out)


def conclusions(h1a_t, h1a_ci, h1b_t, h3_t, st) -> pd.DataFrame:
    rows = []
    drift_scen = "B_drift"
    if len(h1a_ci):
        for (dial, setting, value, ds), g in h1a_ci[h1a_ci.scenario == drift_scen].groupby(
                ["dial", "setting", "value", "dataset"]):
            f, r = g[g.metric == "fpr"].iloc[0], g[g.metric == "recall"].iloc[0]
            rows.append({"dial": dial, "setting": setting, "value": value, "dataset": ds,
                         "claim": "C1 H1a: adaptive FPR < fixed_pooled under drift", "holds": bool(f.ci_high < 0),
                         "evidence": f"FPR diff {100 * f.estimate:+.2f} pts (CI upper {100 * f.ci_high:+.2f})"})
            rows.append({"dial": dial, "setting": setting, "value": value, "dataset": ds,
                         "claim": "C2 H1a: recall non-inferior to fixed_pooled under drift", "holds": bool(r.noninferior),
                         "evidence": f"recall diff {100 * r.estimate:+.2f} pts (CI lower {100 * r.ci_low:+.2f})"})
        t = h1a_t[h1a_t.scenario == drift_scen].set_index(["dial", "setting", "value", "dataset", "arm"]).fpr
        for (dial, setting, value, ds), g in h1a_t[h1a_t.scenario == drift_scen].groupby(["dial", "setting", "value", "dataset"]):
            ad = t.get((dial, setting, value, ds, "adaptive"))
            best = min(t.get((dial, setting, value, ds, m), np.inf) for m in ("ewma", "rolling_mad_cal"))
            rows.append({"dial": dial, "setting": setting, "value": value, "dataset": ds,
                         "claim": "C3 adaptive FPR <= best calibrated rolling baseline under drift", "holds": bool(ad <= best),
                         "evidence": f"adaptive {100 * ad:.2f}% vs best rolling {100 * best:.2f}%"})
    if len(h1b_t):
        p = h1b_t.pivot_table(index=["dial", "setting", "value", "dataset"], columns="arm", values="macro_f1").reset_index()
        fd = h1b_t[h1b_t.arm == "proposed"].set_index(["dial", "setting", "value", "dataset"]).false_drift_stable
        for _, r in p.iterrows():
            k = (r.dial, r.setting, r.value, r.dataset)
            rows.append({"dial": r.dial, "setting": r.setting, "value": r.value, "dataset": r.dataset,
                         "claim": "C4 H1b: proposed macro-F1 >= gated_fixed_pooled",
                         "holds": bool(r.proposed >= r.gated_fixed_pooled - 1e-12),
                         "evidence": f"macro-F1 {r.proposed:.3f} vs {r.gated_fixed_pooled:.3f}; false drift on stable "
                                     f"batches {100 * fd.get(k, np.nan):.1f}%"})
    if len(h3_t):
        for (dial, setting, value, ds), g in h3_t.groupby(["dial", "setting", "value", "dataset"]):
            sus = g[g.scenario.isin(st["h3"]["sustained"])]
            rows.append({"dial": dial, "setting": setting, "value": value, "dataset": ds,
                         "claim": "C5 H3: risk selection useful in sustained degradation (C, D)",
                         "holds": bool(len(sus) and sus.useful.all()),
                         "evidence": f"useful in {int(g.useful.sum())} of {len(g)} scenarios; saving "
                                     f"{g.saving_checks_pct.min():.1f}-{g.saving_checks_pct.max():.1f}% of checks; "
                                     f"lowest post-onset recall (C, D) {sus.post_onset_recall.min():.3f}"})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--dials", nargs="*", default=DIALS, choices=DIALS,
                    help="subset of dials (changing the subset recomputes the cached units)")
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    st = load_settings()
    payload = {"settings": st, "dials": list(args.dials), "driver": file_hash(Path(__file__)),
               "settings_file": file_hash(SETTINGS)}
    raw = run_units(ctx, worker, payload)
    if not len(raw):
        sys.exit("no results")
    crit = ctx.criteria
    out = ctx.out
    ev = raw[raw.evaluation == 1]
    parts, h1a_t, h1a_ci, h1b_t, h3_t = [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    spread = []
    a = ev[ev.family == "h1a"]
    if len(a):
        h1a_t, h1a_ci, per_seed = h1a_tables(a, crit)
        parts.append(h1a_t.assign(family="h1a"))
        h1a_ci.assign(family="h1a").to_csv(out / "sens_h1a_paired_ci.csv", index=False)
        d = per_seed[per_seed.dial == DEFAULT].assign(fpr=lambda x: x.FP / x.N, recall=lambda x: x.TP / x.P)
        for (ds, sc, arm), g in d.groupby(["dataset", "scenario", "arm"]):
            for met in ("fpr", "recall"):
                spread.append({"family": "h1a", "dataset": ds, "scenario": sc, "arm": arm, "metric": met,
                               "mean": g[met].mean(), "sd": g[met].std(ddof=1), "min": g[met].min(),
                               "max": g[met].max(), "seeds": len(g)})
    b = raw[raw.family == "h1b"]
    b = b[b.evaluation == 1]
    if len(b):
        h1b_t, h1b_seed = h1b_tables(b)
        parts.append(h1b_t.assign(family="h1b"))
        for (ds, arm), g in h1b_seed[h1b_seed.dial == DEFAULT].groupby(["dataset", "arm"]):
            spread.append({"family": "h1b", "dataset": ds, "scenario": "A-D", "arm": arm, "metric": "macro_f1",
                           "mean": g.macro_f1.mean(), "sd": g.macro_f1.std(ddof=1), "min": g.macro_f1.min(),
                           "max": g.macro_f1.max(), "seeds": len(g)})
    c = raw[raw.family == "h3"]
    if len(c):
        h3_t = h3_tables(c, ctx.protocol, crit)
        parts.append(h3_t.assign(family="h3", arm="risk"))
    pd.concat(parts, ignore_index=True).to_csv(out / "sens_long.csv", index=False)
    pd.DataFrame(spread).to_csv(out / "sens_seed_spread.csv", index=False)
    concl = conclusions(h1a_t, h1a_ci, h1b_t, h3_t, st)
    concl.to_csv(out / "sens_conclusions.csv", index=False)
    if len(concl):
        summ = concl.groupby(["claim", "dataset"]).agg(
            settings=("holds", "size"), holds=("holds", "sum"),
            holds_at_default=("holds", lambda s: bool(s[concl.loc[s.index, "dial"] == DEFAULT].all())),
            fails_in=("setting", lambda s: "; ".join(f"{concl.loc[i, 'dial']}: {v}" for i, v in s.items()
                                                     if not concl.loc[i, "holds"]))).reset_index()
        summ.to_csv(out / "sens_conclusions_summary.csv", index=False)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 90)
        print(summ.to_string(index=False))
    ctx.manifest({"tables": str(out)}, {"analysis_driver_sha256": payload["driver"],
                                        "analysis_settings_sha256": payload["settings_file"], "dials": list(args.dials),
                                        "note": "post-freeze sensitivity analysis; pipeline code unchanged "
                                                "(code_fingerprint covers the frozen pipeline only)"})


if __name__ == "__main__":
    main()
