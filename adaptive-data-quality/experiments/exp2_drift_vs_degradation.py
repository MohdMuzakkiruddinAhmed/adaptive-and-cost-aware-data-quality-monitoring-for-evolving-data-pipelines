"""Experiment 2: Drift vs. Quality Degradation (project description, Section 13).

Research question  Can the system distinguish legitimate distribution change from actual
                   data-quality degradation?
Scenarios          A stable / no degradation, B drift / no degradation,
                   C no drift / degradation, D drift + degradation (flagship).
Method             Each batch gets two signals:
                     drift       any column's drift score (vs. the warm-up baseline) above its threshold
                     degradation adaptive rule-failure rate above its threshold
                   Thresholds = mean + 3 sd of the same signals on the warm-up batches T1..T5
                   (leave-one-out, so calibration batches have full history).
                   Batch class = stable | drift | degradation | drift+degradation.
Baseline           Static monitoring: static rule-failure rate above its own warm-up
                   threshold -> "degradation" (static monitoring has no notion of drift).
Measures           Confusion matrices, drift classification, defect detection, false alarms,
                   missed defects. Evaluation batches T6..T20.

    python -m experiments.exp2_drift_vs_degradation
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import datasets_config, experiment_config, results_dir, write_json  # noqa: E402
from experiments.detection import run_all, warmup_calibration  # noqa: E402
from experiments.reporting import COLORS, DATASET_LABELS, aggregate, style_axes, write_run_log  # noqa: E402

CLASSES = ["stable", "drift", "degradation", "drift+degradation"]


def to_class(drift: bool, degr: bool) -> str:
    return CLASSES[int(drift) + 2 * int(degr)]


def classify(batch: pd.DataFrame, datasets, exp, z: float | None = None):
    """Calibrate the two thresholds per dataset on the warm-up batches (leave-one-out, mean + z sd)
    and classify every evaluation batch. Returns (per-batch predictions, thresholds)."""
    z = exp["classification"]["calibration_z"] if z is None else z
    thresholds, preds = {}, []
    for ds, cfg in datasets.items():
        cal = warmup_calibration(ds, "A_stable", cfg, exp)   # T1..T5 are identical in every scenario
        dcols = [c for c in cal.columns if c.startswith("D_")]
        thr = {c: float(cal[c].mean() + z * cal[c].std(ddof=1)) for c in dcols}
        thr["failure_rate"] = float(cal.failure_rate.mean() + z * cal.failure_rate.std(ddof=1))
        thr["static_failure_rate"] = float(cal.static_failure_rate.mean() + z * cal.static_failure_rate.std(ddof=1))
        thresholds[ds] = {"thresholds": thr, "calibration": cal.to_dict(orient="records")}

        ev = batch[(batch.dataset == ds) & (batch.evaluation == 1)]
        for (scen, b), g in ev.groupby(["scenario", "batch"]):
            a = g[g.method == "adaptive"].iloc[0]
            s = g[g.method == "static"].iloc[0]
            exceed = {c: a[c] > thr[c] for c in dcols}
            true = to_class(bool(a.drift_label), bool(a.degradation_label))
            preds.append({"dataset": ds, "scenario": scen, "batch": b, "drift_level": a.drift_level,
                          "defect_rate": a.defect_rate, "true_class": true,
                          "drift_signal": max(a[c] / thr[c] for c in dcols),
                          "drifted_columns": ",".join(c[2:] for c, v in exceed.items() if v),
                          "failure_rate": a.failure_rate, "static_failure_rate": s.failure_rate,
                          "adaptive_class": to_class(any(exceed.values()), a.failure_rate > thr["failure_rate"]),
                          "static_class": to_class(False, s.failure_rate > thr["static_failure_rate"])})
    return pd.DataFrame(preds), thresholds


def summarise_classification(pred: pd.DataFrame, datasets):
    """Confusion matrices and accuracy / detection / false-alarm rates, per dataset and overall."""
    cm_rows, acc_rows = [], []
    for ds in list(datasets) + ["ALL"]:
        p = pred if ds == "ALL" else pred[pred.dataset == ds]
        for m in ("adaptive", "static"):
            cm = pd.crosstab(pd.Categorical(p.true_class, CLASSES), pd.Categorical(p[f"{m}_class"], CLASSES),
                             dropna=False)
            for tc in CLASSES:
                cm_rows.append({"dataset": ds, "method": m, "true_class": tc, **{f"pred_{pc}": int(cm.loc[tc, pc]) for pc in CLASSES}})
            drift_true = p.true_class.str.startswith("drift")
            degr_true = p.true_class.str.contains("degradation")
            drift_pred = p[f"{m}_class"].str.startswith("drift")
            degr_pred = p[f"{m}_class"].str.contains("degradation")
            acc_rows.append({
                "dataset": ds, "method": m, "batches": len(p),
                "accuracy_4class": float((p.true_class == p[f"{m}_class"]).mean()),
                "drift_detection_rate": float(drift_pred[drift_true].mean()),
                "drift_false_alarm_rate": float(drift_pred[~drift_true].mean()),
                "degradation_detection_rate": float(degr_pred[degr_true].mean()),
                "degradation_false_alarm_rate": float(degr_pred[~degr_true].mean()),
                "false_degradation_alarms_during_pure_drift": float(degr_pred[p.true_class == "drift"].mean()),
            })
    return pd.DataFrame(cm_rows), pd.DataFrame(acc_rows)


def main() -> None:
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    batch = run_all(datasets, exp)
    raw = results_dir("raw", "exp2_batch_results.csv"); raw.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw, index=False)
    pred, thresholds = classify(batch, datasets, exp)

    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    pred.to_csv(tables / "exp2_batch_classification.csv", index=False)
    write_json(thresholds, results_dir("raw", "exp2_thresholds.json"))

    cm_df, acc = summarise_classification(pred, datasets)
    cm_df.to_csv(tables / "exp2_confusion_matrices.csv", index=False)
    acc.to_csv(tables / "exp2_classification_summary.csv", index=False)

    # record-level: defects detected, false alarms, missed defects per scenario
    ev = batch[batch.evaluation == 1]
    rec = aggregate(ev, ["dataset", "scenario", "method"])
    rec = rec.rename(columns={"TP": "defects_detected", "FP": "false_alarms", "FN": "missed_defects"})
    rec_cols = ["dataset", "scenario", "method", "defects", "defects_detected", "missed_defects", "false_alarms",
                "precision", "recall", "f1", "fpr"]
    rec[rec_cols].to_csv(tables / "exp2_record_level_by_scenario.csv", index=False)

    # figure: the two signals in the flagship scenario D
    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharex=True)
    for j, ds in enumerate(datasets):
        p = pred[(pred.dataset == ds) & (pred.scenario == "D_drift_degradation")].sort_values("batch")
        t = p.batch.str[1:].astype(int)
        ax = axes[0, j]
        ax.plot(t, p.drift_signal, color=COLORS["adaptive"], marker="o", markersize=4, linewidth=2)
        ax.axhline(1.0, color="#55554f", linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(DATASET_LABELS[ds], fontsize=10, loc="left", color="#1f1f1c")
        style_axes(ax)
        ax = axes[1, j]
        ax.plot(t, 100 * p.failure_rate, color=COLORS["adaptive"], marker="o", markersize=4, linewidth=2)
        ax.axhline(100 * thresholds[ds]["thresholds"]["failure_rate"], color="#55554f", linewidth=1, linestyle="--")
        ax.set_xlabel("Batch (T)", fontsize=9, color="#55554f")
        style_axes(ax)
    axes[0, 0].set_ylabel("Drift signal\n(score / threshold, log)", fontsize=9, color="#55554f")
    axes[1, 0].set_ylabel("Rule failure rate (%)", fontsize=9, color="#55554f")
    fig.text(0.99, 0.01, "Scenario D: drift ramps 2%->30% from T6; defects 1%->3%->5%->10% from T11. Dashed: threshold.",
             ha="right", fontsize=8, color="#55554f")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(figs / "exp2_signals_scenario_D.png", dpi=300)
    plt.close(fig)

    summary = {f"{r.dataset}|{r.method}": {k: getattr(r, k) for k in acc.columns if k not in ("dataset", "method")}
               for r in acc.itertuples()}
    log = write_run_log("exp2_drift_vs_degradation", datasets, exp, batch, summary,
                        {"batch_results": str(raw), "tables": str(tables), "figures": str(figs)},
                        time.perf_counter() - t0)
    pd.set_option("display.width", 220)
    print(acc.round(3).to_string(index=False))
    print()
    print(cm_df[cm_df.dataset == "ALL"].to_string(index=False))
    print(f"\nRun log: {log}")


if __name__ == "__main__":
    main()
