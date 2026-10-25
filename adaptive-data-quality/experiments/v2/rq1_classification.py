"""RQ1 / H1b: distinguishing legitimate drift from degradation (supervisor experiment 2).

Classes      stable, legitimate drift, degradation, both (scenarios A-D).
Comparators  protocol.yaml classification.comparators, each = drift signal x quality-failure signal:
               static_no_drift      PoC comparator (no drift output: structural 50% ceiling), kept as weak baseline
               proposed             gated drift + adaptive failure rate
               proposed_no_gate     ungated drift + adaptive failure rate (error-filtering gate ablation)
               proposed_min_effect  proposed + a practical-significance floor on the drift scores (sensitivity)
               gated_fixed_pooled   SAME gated drift detector, failure rate of pooled fixed rules
               gated_rolling_mad    SAME gated drift detector, failure rate of rolling MAD rules
               gated_ewma           SAME gated drift detector, failure rate of EWMA rules
               avh_style            AVH-style batch metrics mapped to drift / degradation (baselines/avh.py),
                                    history of all earlier batches (as published)
               avh_style_quarantine the same, learning only from batches without an alarm
               deequ_anomaly        Amazon Deequ's anomaly checks on its batch metrics, as documented (OnlineNormal,
                                    cumulative history; baselines/external_tools.py)
               deequ_anomaly_calibrated  the same checks with the warm-up as reference and the deviation factor
                                    calibrated like the proposed method's thresholds
               deequ_calibrated_checks   + a failed suggested constraint (Deequ verification) = degradation
               tfdv_*               TensorFlow Data Validation: drift comparators + schema anomalies, against the
                                    previous batch (as documented) or the warm-up, documented or calibrated thresholds
                                    (Deequ and TFDV run as sidecars: experiments/v2/tool_exchange.py; comparators
                                    whose outputs are missing are skipped with a warning)
             Holding the drift detector constant while changing the failure signal isolates the failure
             signal; proposed vs proposed_no_gate isolates the gate.
Thresholds   leave-one-out over the warm-up, identical procedure for every monitor-based comparator
             (drift: protocol.yaml classification.drift_threshold; failure rate: mean + z sd).
Measures     four-class macro-F1, balanced accuracy, accuracy, confusion matrices, false degradation alarms
             during pure drift, drift / degradation detection delay (batches after onset), insufficient-
             evidence share, gate excluded fraction; paired seed-level CIs of macro-F1 and balanced accuracy
             against "proposed".

    python -m experiments.v2.rq1_classification --split dev --jobs 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.avh import AVHMonitor  # noqa: E402
from evaluation.metrics import confusion, label_metrics, onset_delay  # noqa: E402
from evaluation.stats import holm, paired_mean_diff  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, true_class)
from monitor.classifier import to_class  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402

EXPERIMENT = "rq1_classification"
DRIFT_MAP = {"none": "none", "gated": "gated", "ungated": "ungated"}


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        for name, spec in payload["comparators"].items():
            if spec["drift"] in ("deequ", "tfdv"):
                labels = _tool_labels(spec, ds, seed, scen, payload)
                if labels is None:
                    continue
                for t in range(1, rep.replay.n_batches + 1):
                    b = rep.batch(t)
                    lab = labels.get(t, {"drift": False, "degradation": False, "fired": "", "not_ready": ""})
                    rows.append({**b.meta, "dataset": ds, "seed": seed, "scenario": scen, "comparator": name,
                                 "pred_label": to_class(lab["drift"], lab["degradation"]),
                                 "pred_drift": lab["drift"], "pred_degradation": lab["degradation"],
                                 "tool_fired": lab["fired"], "tool_not_ready": lab["not_ready"]})
                continue
            if spec["drift"] == "avh":
                a = pr["avh"]
                avh = AVHMonitor(cfg["numeric"], cfg["categorical"], a["fpr_budget"], a["min_history"],
                                 a.get("bound", "student"), spec.get("history", a.get("history", "all")))
                for t, b, (prof, _) in rep:
                    res = avh.observe(prof)
                    rows.append({**b.meta, "dataset": ds, "seed": seed, "scenario": scen, "comparator": name,
                                 "pred_label": to_class(res["drift"], res["degradation"]),
                                 "pred_drift": res["drift"], "pred_degradation": res["degradation"],
                                 "avh_fired": ",".join(res["fired"]), "avh_ready": res["ready"]})
                continue
            dt = None
            if spec.get("min_effect"):                   # sensitivity: drift must also exceed a practical floor
                dt = dict(pr["classification"].get("drift_threshold", {}),
                          min_effect=pr["classification"]["practical_min_effect"])
            mc = MonitorConfig(name=name, method=spec["failure"], drift_source=DRIFT_MAP[spec["drift"]],
                               degradation_rule="union_rate", scheduler="all", drift_threshold=dt)
            m = Monitor(cfg, exp, pr, mc, seed=seed)
            rows += run_monitor_over(rep, m, extra={"comparator": name})
    return pd.DataFrame(rows)


def _tool_labels(spec, ds, seed, scen, payload):
    from baselines.external_tools import deequ_flags, deequ_labels, load_series, load_verify, tfdv_labels
    exchange = Path(payload["exchange"])
    series = load_series(exchange, spec["drift"], ds, seed, scen)
    if series is None:
        return None
    if spec["drift"] == "deequ":
        cf = None
        if spec.get("constraints"):
            ver = load_verify(exchange, ds, seed, scen)
            if ver is None:
                return None
            cf = {t: f["failed"] for t, f in deequ_flags(ver, spec["constraints"], "failed").items()}
        return deequ_labels(series, spec.get("config", "default"), cf)
    return tfdv_labels(series, spec.get("reference", "warmup"), spec.get("thresholds", "documented"),
                       payload["tfdv_documented"], payload["alpha"])


def tool_payload(ctx, comps):
    """Exchange location and a fingerprint of the tool outputs (new outputs -> recompute); drops tool comparators
    whose outputs do not exist, with a warning, so the experiment runs without the sidecars too."""
    from experiments.v2.tool_exchange import exchange_dir, missing_outputs, tool_outputs_digest
    root = exchange_dir(ctx.split)
    keep = {}
    for name, spec in comps.items():
        if spec["drift"] in ("deequ", "tfdv"):
            miss = missing_outputs(ctx.split, spec["drift"], list(ctx.datasets), ctx.seeds, ctx.scenarios)
            if spec.get("constraints") and miss == 0:
                miss = missing_outputs(ctx.split, "deequ_verify", list(ctx.datasets), ctx.seeds, ctx.scenarios)
            if miss is None or miss > 0:
                why = "batches not exported" if miss is None else f"{miss} run(s) without current {spec['drift']} output"
                print(f"[{EXPERIMENT}] WARNING: comparator {name} skipped ({why}; see "
                      f"tools/external_validators/README.md)", flush=True)
                continue
        keep[name] = spec
    tcfg = ctx.protocol.get("external_tools", {})
    return keep, {"exchange": str(root), "tool_outputs": tool_outputs_digest(ctx.split),
                  "tfdv_documented": tcfg.get("tfdv", {}).get("documented_thresholds", {}),
                  "alpha": ctx.protocol["classification"].get("drift_threshold", {}).get("alpha", 0.05)}


def main(argv=None) -> None:
    args = parser(__doc__).parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    comps, tools = tool_payload(ctx, ctx.protocol["classification"]["comparators"])
    raw = run_units(ctx, worker, {"comparators": comps, **tools})
    ev = raw[raw.evaluation == 1].copy()
    ev["true_class"] = ev.apply(true_class, axis=1)
    crit = ctx.criteria

    summ, cms, delays, per_seed = [], [], [], []
    for (ds, comp), g in ev.groupby(["dataset", "comparator"]):
        summ.append({"dataset": ds, "comparator": comp, "seeds": g.seed.nunique(),
                     **label_metrics(g.true_class, g.pred_label),
                     "gate_excluded_fraction": g["gate_excluded_fraction"].mean() if "gate_excluded_fraction" in g else np.nan})
        cm = confusion(g.true_class, g.pred_label).reset_index().rename(columns={"index": "true_class"})
        cms.append(cm.assign(dataset=ds, comparator=comp))
        for s, gs in g.groupby("seed"):
            per_seed.append({"dataset": ds, "comparator": comp, "seed": s, **label_metrics(gs.true_class, gs.pred_label)})
    for (ds, comp, s, sc), g in raw.groupby(["dataset", "comparator", "seed", "scenario"]):
        d_on = int(g.loc[g.drift_level > 0, "t"].min()) if (g.drift_level > 0).any() else None
        g_on = int(g.loc[g.degradation_label == 1, "t"].min()) if (g.degradation_label == 1).any() else None
        ge = g[g.evaluation == 1]
        delays.append({"dataset": ds, "comparator": comp, "seed": s, "scenario": sc,
                       "drift_delay": onset_delay(ge.t, ge.pred_drift.fillna(False).astype(bool), d_on),
                       "degradation_delay": onset_delay(ge.t, ge.pred_degradation.fillna(False).astype(bool), g_on)})
    pd.DataFrame(summ).to_csv(ctx.out / "rq1b_classification_summary.csv", index=False)
    pd.concat(cms, ignore_index=True).to_csv(ctx.out / "rq1b_confusion_matrices.csv", index=False)
    dl = pd.DataFrame(delays)
    dl.to_csv(ctx.out / "rq1b_delays_per_run.csv", index=False)
    dl.groupby(["dataset", "comparator", "scenario"])[["drift_delay", "degradation_delay"]].agg(["mean", "max", "count"]) \
        .to_csv(ctx.out / "rq1b_delays.csv")

    ps = pd.DataFrame(per_seed)
    rows = []
    for ds, g in ps.groupby("dataset"):
        ref = g[g.comparator == "proposed"].set_index("seed")
        for comp, gc in g.groupby("comparator"):
            if comp == "proposed":
                continue
            gc = gc.set_index("seed")
            common = sorted(set(gc.index) & set(ref.index))
            for metric in ("macro_f1", "balanced_accuracy", "false_degradation_during_drift"):
                r = paired_mean_diff(ref.loc[common, metric], gc.loc[common, metric], crit["bootstrap_replicates"],
                                     crit["confidence_level"])
                rows.append({"dataset": ds, "proposed_minus": comp, "metric": metric, **r})
    ci = pd.DataFrame(rows)
    if len(ci):
        ci["p_holm"] = np.nan
        for _, idx in ci.groupby(["dataset", "metric"]).groups.items():
            ci.loc[idx, "p_holm"] = holm(ci.loc[idx, "p_value"].fillna(1).to_numpy())
    ci.to_csv(ctx.out / "rq1b_paired_ci.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"comparators": comps})
    print(pd.DataFrame(summ)[["dataset", "comparator", "accuracy", "macro_f1", "balanced_accuracy",
                              "false_degradation_during_drift"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
