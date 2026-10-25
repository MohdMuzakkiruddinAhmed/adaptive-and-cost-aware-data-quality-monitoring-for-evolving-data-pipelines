"""Experiment 4: Cost-Aware Rule Execution (project description, Section 13).

Research question  Can risk-aware selective execution reduce computation cost without
                   significantly reducing detection?
Baseline           Run-all: adaptive thresholds, every check on every batch.
Proposed method    Risk score R_i (Section 11) + cost-aware selector (Section 12): basic checks
                   always, standard checks (category, uniqueness) only for units at moderate
                   risk or above, complete suite when any unit is very high, refresh every 5 batches.
Controlled vars.   Same batches, defects, adaptive thresholds, seed.
Dependent vars.    Precision, recall, F1, defects detected, missed defects, rules executed,
                   runtime, data scanned, cost saving.
                   CostSaving = (C_baseline - C_proposed) / C_baseline x 100
Extra              Sweep of the moderate-risk threshold (Section 12: "starting points only and
                   should be tuned experimentally") -> cost vs detection curve.

    python -m experiments.exp4_cost_optimization
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import datasets_config, experiment_config, results_dir  # noqa: E402
from experiments.cost_aware import run  # noqa: E402
from experiments.reporting import DATASET_LABELS, style_axes, write_run_log  # noqa: E402

DATASET_COLORS = {"bts_airline": "#2a78d6", "online_retail": "#eb6834", "chicago_food": "#1baf7a"}
DATASET_MARKERS = {"bts_airline": "o", "online_retail": "s", "chicago_food": "^"}


def summarise(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by).agg(defects=("defects", "sum"), TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
                           TN=("TN", "sum"), rules_executed=("rules_executed", "sum"),
                           rules_available=("rules_available", "sum"), runtime_ms=("runtime_ms", "sum"),
                           bytes_scanned=("bytes_scanned", "sum")).reset_index()
    g["precision"] = g.TP / (g.TP + g.FP)
    g["recall"] = g.TP / (g.TP + g.FN)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g["fpr"] = g.FP / (g.FP + g.TN)
    g["data_scanned_mb"] = g.bytes_scanned / 1e6
    return g.rename(columns={"TP": "defects_detected", "FN": "missed_defects", "FP": "false_alarms"})


def add_savings(s: pd.DataFrame, keys) -> pd.DataFrame:
    base = s[s["mode"] == "run_all"].set_index(keys)
    s = s.copy()
    for col, name in (("bytes_scanned", "saving_data_pct"), ("runtime_ms", "saving_runtime_pct"),
                      ("rules_executed", "saving_rules_pct")):
        ref = s.set_index(keys).index.map(base[col])
        s[name] = 100 * (ref.to_numpy() - s[col].to_numpy()) / ref.to_numpy()
    return s


def main() -> None:
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    rows = []
    for ds, cfg in datasets.items():
        for sc in exp["scenarios"]:
            rows += run(ds, sc, cfg, exp, "run_all")
            rows += run(ds, sc, cfg, exp, "selective")
            for m in exp["risk"]["sweep_moderate"]:
                rows += [dict(r, mode="sweep") for r in run(ds, sc, cfg, exp, "selective", moderate=m)]
    batch = pd.DataFrame(rows)
    raw = results_dir("raw", "exp4_batch_results.csv"); raw.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw, index=False)
    ev = batch[batch.evaluation == 1]

    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    main_ev = ev[ev["mode"].isin(["run_all", "selective"])]
    per = add_savings(summarise(main_ev, ["dataset", "scenario", "mode"]), ["dataset", "scenario"])
    cols = ["dataset", "scenario", "mode", "precision", "recall", "f1", "defects", "defects_detected",
            "missed_defects", "false_alarms", "rules_executed", "runtime_ms", "data_scanned_mb",
            "saving_data_pct", "saving_runtime_pct", "saving_rules_pct"]
    per[cols].to_csv(tables / "exp4_cost_aware.csv", index=False)
    overall = add_savings(summarise(main_ev, ["scenario", "mode"]), ["scenario"])
    overall[[c for c in cols if c != "dataset"]].to_csv(tables / "exp4_cost_aware_by_scenario.csv", index=False)

    # threshold sweep: cost vs detection over all scenarios
    sw = ev[ev["mode"] == "sweep"]
    ref = summarise(ev[ev["mode"] == "run_all"], ["dataset"]).set_index("dataset")
    curve = summarise(sw, ["dataset", "moderate_threshold"])
    curve["saving_data_pct"] = 100 * (1 - curve.bytes_scanned / curve.dataset.map(ref.bytes_scanned))
    curve["saving_runtime_pct"] = 100 * (1 - curve.runtime_ms / curve.dataset.map(ref.runtime_ms))
    curve["recall_run_all"] = curve.dataset.map(ref.recall)
    curve[["dataset", "moderate_threshold", "recall", "recall_run_all", "precision", "f1", "saving_data_pct",
           "saving_runtime_pct", "rules_executed"]].to_csv(tables / "exp4_threshold_sweep.csv", index=False)

    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    spec = exp["risk"]["thresholds"]["moderate"]
    for ds in datasets:
        c = curve[curve.dataset == ds].sort_values("moderate_threshold")
        ax.plot(c.saving_data_pct, c.recall, color=DATASET_COLORS[ds], marker=DATASET_MARKERS[ds], markersize=6,
                linewidth=2, label=DATASET_LABELS[ds])
        s = c[c.moderate_threshold == spec]
        ax.scatter(s.saving_data_pct, s.recall, s=140, facecolors="none", edgecolors="#1f1f1c", linewidths=1.2, zorder=5)
    ax.set_xlabel("Data scanned saved vs run-all (%)", fontsize=9, color="#55554f")
    ax.set_ylabel("Recall", fontsize=9, color="#55554f")
    ax.set_xlim(left=-2)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(0.99, 0.01, f"Points: moderate-risk threshold {exp['risk']['sweep_moderate'][0]} to "
             f"{exp['risk']['sweep_moderate'][-1]}; circled = Section 12 value ({spec}). All scenarios, T6-T20.",
             ha="right", fontsize=7.5, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / "exp4_cost_vs_detection.png", dpi=300)
    plt.close(fig)

    sel = per[per["mode"] == "selective"]
    summary = {f"{r.dataset}|{r.scenario}": {"recall": r.recall, "saving_data_pct": r.saving_data_pct,
                                             "saving_runtime_pct": r.saving_runtime_pct} for r in sel.itertuples()}
    log = write_run_log("exp4_cost_optimization", datasets, exp, batch[batch["mode"] != "sweep"], summary,
                        {"batch_results": str(raw), "tables": str(tables), "figures": str(figs)},
                        time.perf_counter() - t0)
    pd.set_option("display.width", 220)
    show = ["dataset", "scenario", "mode", "precision", "recall", "f1", "missed_defects", "rules_executed",
            "runtime_ms", "data_scanned_mb", "saving_data_pct", "saving_runtime_pct"]
    print(per[show].round(3).to_string(index=False))
    print()
    print(overall[[c for c in show if c != "dataset"]].round(3).to_string(index=False))
    print()
    print(curve[["dataset", "moderate_threshold", "recall", "saving_data_pct", "saving_runtime_pct"]].round(3).to_string(index=False))
    print(f"\nRun log: {log}")


if __name__ == "__main__":
    main()
