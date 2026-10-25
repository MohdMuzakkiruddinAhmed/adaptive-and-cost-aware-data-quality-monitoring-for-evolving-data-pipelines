"""Experiment 1: Static vs. Adaptive Monitoring (project description, Section 13).

Research question  Does adaptive monitoring reduce false alarms when legitimate data
                   distributions change?
Hypothesis         During legitimate drift, adaptive thresholds give a lower false-positive
                   rate than static thresholds without materially reducing recall.
Baseline           Static thresholds: rules fixed from the T1 profile.
Proposed method    Adaptive thresholds T_t = f(H_{t-k:t}, D_t).
Independent var.   Monitoring method; drift level (scenario A: none, B: gradual 2% -> 30%).
Controlled vars.   Same batches, same injected defects (1% of records, 4 types), same rule
                   types, k = 3, same seed.
Dependent vars.    Precision, recall, F1, false-positive rate, false-negative rate.
Evaluation         Batches T6..T20 (T1..T5 are warm-up history), micro-averaged.

    python -m experiments.exp1_static_vs_adaptive
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
from experiments.detection import run_all  # noqa: E402
from experiments.reporting import (COLORS, DATASET_LABELS, LABELS, MARKERS, aggregate, style_axes,  # noqa: E402
                                   write_run_log)

SCENARIOS = ["A_stable", "B_drift"]


def figure(batch: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    ds_list = list(DATASET_LABELS)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(12, 3.4), sharex=True)
    b = batch[(batch.scenario == "B_drift") & (batch.evaluation == 1)]
    for ax, ds in zip(axes, ds_list):
        for m in ("static", "adaptive"):
            s = b[(b.dataset == ds) & (b.method == m)].sort_values("drift_level")
            ax.plot(100 * s.drift_level, s[metric], color=COLORS[m], marker=MARKERS[m], markersize=5,
                    linewidth=2, label=LABELS[m])
        ax.set_title(DATASET_LABELS[ds], fontsize=10, color="#1f1f1c", loc="left")
        ax.set_xlabel("Drift level (%)", fontsize=9, color="#55554f")
        ax.set_ylim(bottom=0)
        style_axes(ax)
    axes[0].set_ylabel(ylabel, fontsize=9, color="#55554f")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    batch = run_all(datasets, exp)
    raw = results_dir("raw", "exp1_batch_results.csv"); raw.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw, index=False)

    ev = batch[(batch.evaluation == 1) & batch.scenario.isin(SCENARIOS)]
    table = aggregate(ev, ["dataset", "scenario", "method"])
    cols = ["dataset", "scenario", "method", "precision", "recall", "f1", "fpr", "fnr", "TP", "FP", "FN", "TN"]
    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    table[cols].to_csv(tables / "exp1_static_vs_adaptive.csv", index=False)

    # recall by defect type (and outlier severity) during drift: "without materially reducing recall"
    b = ev[ev.scenario == "B_drift"]
    rec = []
    for (ds, m), g in b.groupby(["dataset", "method"]):
        row = {"dataset": ds, "method": m}
        for t in ["missing", "duplicate", "invalid_category", "outlier", "outlier_Medium", "outlier_High"]:
            row[f"recall_{t}"] = g[f"tp_{t}"].sum() / g[f"n_{t}"].sum() if g[f"n_{t}"].sum() else None
        rec.append(row)
    pd.DataFrame(rec).to_csv(tables / "exp1_recall_by_defect_type_drift.csv", index=False)

    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    figure(batch, "fpr", "False-positive rate", figs / "exp1_fpr_vs_drift.png")
    figure(batch, "f1", "F1-score", figs / "exp1_f1_vs_drift.png")

    summary = {f"{r.dataset}|{r.scenario}|{r.method}": {k: getattr(r, k) for k in ("precision", "recall", "f1", "fpr", "fnr")}
               for r in table.itertuples()}
    log = write_run_log("exp1_static_vs_adaptive", datasets, exp, batch, summary,
                        {"batch_results": str(raw), "table": str(tables / "exp1_static_vs_adaptive.csv"),
                         "figures": str(figs)}, time.perf_counter() - t0)

    pd.set_option("display.width", 200)
    print(table[cols].round(4).to_string(index=False))
    print(f"\nRun log: {log}")


if __name__ == "__main__":
    main()
