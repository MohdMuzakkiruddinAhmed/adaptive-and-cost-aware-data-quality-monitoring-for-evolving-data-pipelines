"""Experiment 5: Ablation Study (project description, Section 13).

Research question  Which components are responsible for performance improvements?
Configurations     1 Static                  static thresholds (T1), every check
                   2 LLM only                validated LLM rules only (Experiment 3), every check
                   3 Adaptive                adaptive thresholds, every check
                   4 Adaptive + Cost         adaptive thresholds, risk-based selective execution
                   5 Full proposed system    adaptive + cost-aware selection + validated LLM rules
                   Configurations 2 and 5 use the validated rules of every Experiment 3 run and
                   report the mean over runs; they stay "pending" until Experiment 3 has run.
Measures           Precision, recall, F1, false-positive rate, false-negative rate, runtime,
                   rules executed, cost (data scanned). Evaluation batches T6..T20 of all four
                   scenarios, micro-averaged.

    python -m experiments.exp5_ablation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import datasets_config, experiment_config, results_dir  # noqa: E402
from experiments.cost_aware import run  # noqa: E402
from experiments.exp4_cost_optimization import summarise  # noqa: E402
from experiments.reporting import DATASET_LABELS, style_axes, write_run_log  # noqa: E402

CONFIGS = [  # (name, mode, uses LLM rules)
    ("1 Static", "static", False),
    ("2 LLM only", "fixed", True),
    ("3 Adaptive", "run_all", False),
    ("4 Adaptive + Cost", "selective", False),
    ("5 Full proposed system", "selective", True),
]
CONFIG_COLORS = {"1 Static": "#eb6834", "2 LLM only": "#e87ba4", "3 Adaptive": "#2a78d6",
                 "4 Adaptive + Cost": "#1baf7a", "5 Full proposed system": "#eda100"}


def load_llm_rules(ds: str, mock: bool):
    """Validated LLM rules of every Experiment 3 run for a dataset: {run: rules}."""
    d = results_dir("llm_rules_mock" if mock else "llm_rules") / ds
    out = {}
    for p in sorted(d.glob("run_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[int(rec["run"])] = rec["validated_rules"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use rules from the offline Experiment 3 test")
    args, _ = ap.parse_known_args()
    t0 = time.perf_counter()
    sfx = "_mock" if args.mock else ""
    datasets, exp = datasets_config(), experiment_config()
    llm_rules = {ds: load_llm_rules(ds, args.mock) for ds in datasets}
    have_llm = all(llm_rules[ds] for ds in datasets)
    rows = []
    for name, mode, uses_llm in CONFIGS:
        if uses_llm and not have_llm:
            continue
        for ds, cfg in datasets.items():
            runs = llm_rules[ds] if uses_llm else {0: None}
            for r, extra in runs.items():
                for sc in exp["scenarios"]:
                    rows += [dict(x, configuration=name, run=r) for x in run(ds, sc, cfg, exp, mode, extra_rules=extra)]
    batch = pd.DataFrame(rows)
    raw = results_dir("raw", f"exp5_batch_results{sfx}.csv"); raw.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw, index=False)
    ev = batch[batch.evaluation == 1]

    def mean_over_runs(by):
        s = summarise(ev, by + ["run"])
        s["fnr"] = s.missed_defects / (s.missed_defects + s.defects_detected)
        num = [c for c in s.columns if c not in by + ["run"]]
        m = s.groupby(by)[num].mean().reset_index()
        m["runs"] = s.groupby(by)["run"].nunique().to_numpy()
        return m

    per_ds = mean_over_runs(["configuration", "dataset"])
    overall = mean_over_runs(["configuration"]).assign(dataset="ALL")
    per_sc = mean_over_runs(["configuration", "scenario"])
    table = pd.concat([per_ds, overall], ignore_index=True)
    # pending rows (LLM configurations) so the table has the full Section 13 structure
    pend = [] if have_llm else [{"configuration": n, "dataset": d, "status": "pending: run Experiment 3"}
                                for n, m, u in CONFIGS if u for d in list(datasets) + ["ALL"]]
    table["status"] = "done"
    table = pd.concat([table, pd.DataFrame(pend)], ignore_index=True)
    order = {n: i for i, (n, _, _) in enumerate(CONFIGS)}
    table = table.sort_values(["dataset", "configuration"], key=lambda s: s.map(order) if s.name == "configuration" else s)
    cols = ["dataset", "configuration", "status", "runs", "precision", "recall", "f1", "fpr", "fnr", "rules_executed",
            "runtime_ms", "data_scanned_mb"]
    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    table[cols].to_csv(tables / f"exp5_ablation{sfx}.csv", index=False)
    per_sc[["configuration", "scenario", "precision", "recall", "f1", "fpr", "rules_executed", "runtime_ms",
            "data_scanned_mb"]].to_csv(tables / f"exp5_ablation_by_scenario{sfx}.csv", index=False)

    # Figure 7: ablation comparison (F1, false-positive rate, data scanned) per dataset
    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    done = [n for n, m, u in CONFIGS if have_llm or not u]
    ds_list = list(datasets)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    width = 0.8 / len(done)
    x = np.arange(len(ds_list))
    for ax, (metric, label) in zip(axes, [("f1", "F1-score"), ("fpr", "False-positive rate"),
                                          ("data_scanned_mb", "Data scanned (MB)")]):
        for j, n in enumerate(done):
            vals = [per_ds[(per_ds.configuration == n) & (per_ds.dataset == d)][metric].iloc[0] for d in ds_list]
            ax.bar(x + (j - (len(done) - 1) / 2) * width, vals, width * 0.92, color=CONFIG_COLORS[n], label=n[2:])
        ax.set_xticks(x, [DATASET_LABELS[d].replace(" Inspections", "") for d in ds_list], fontsize=8.5)
        ax.set_title(label, fontsize=10, loc="left", color="#1f1f1c")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    note = "All four scenarios, T6-T20; LLM configurations: mean over Experiment 3 runs." if have_llm else \
        "LLM only and Full proposed system pending (Experiment 3). All four scenarios, T6-T20."
    fig.text(0.99, 0.01, note + (" MOCK LLM." if args.mock else ""), ha="right", fontsize=8, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figs / ("exp5_ablation_mock.png" if args.mock else "exp5_ablation.png"), dpi=300)
    plt.close(fig)

    summary = {f"{r.dataset}|{r.configuration}": {k: getattr(r, k) for k in ("precision", "recall", "f1", "fpr")}
               for r in table[table.status == "done"].itertuples()}
    log = write_run_log(f"exp5_ablation{sfx}", datasets, exp, batch, summary,
                        {"batch_results": str(raw), "table": str(tables / f"exp5_ablation{sfx}.csv"), "figures": str(figs)},
                        time.perf_counter() - t0)
    pd.set_option("display.width", 220)
    print(table[cols].round(4).to_string(index=False))
    print(f"\nRun log: {log}")


if __name__ == "__main__":
    main()
