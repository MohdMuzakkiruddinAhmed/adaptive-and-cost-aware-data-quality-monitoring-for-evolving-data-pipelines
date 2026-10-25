"""Experiment 3b: LLM rules on semantic defects (extension of Experiment 3, Sections 6, 8.3, 13).

Experiment 3 showed that validated LLM rules add no coverage on the standard injected defects,
which are statistically obvious. This experiment asks whether they add coverage on realistic
domain violations that statistical rules cannot see (configs/datasets.yaml: semantic_defects):
values that are impossible in the domain but not statistically unusual (a negative taxi-out
time, a zero air time, a negative price) and plausible-looking invalid values (a well-formed but
non-existent code, a misspelled category).

Scenarios      S1 stable / S2 gradual drift: same batches, drift and 1% standard defects as
               scenarios A / B, plus semantic defects in 1% of records from T6 on (absent
               from the warm-up, so no method can learn them as normal).
Methods        Static rules | Adaptive | LLM + validation | Static + validated LLM |
               Adaptive + validated LLM. The LLM rules are the saved Experiment 3 rules (no new
               LLM calls); LLM configurations are the mean over the Experiment 3 runs.
Measures       Recall on semantic defects (overall and per violated column), recall on standard
               defects, precision, F1, false-positive rate. T6..T20.

    python -m experiments.exp3b_semantic_defects
"""
from __future__ import annotations

import argparse
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
from experiments.exp5_ablation import load_llm_rules  # noqa: E402
from experiments.reporting import DATASET_LABELS, style_axes, write_run_log  # noqa: E402

CONFIGS = [  # (name, mode, uses validated LLM rules)
    ("Static rules", "static", False),
    ("Adaptive", "run_all", False),
    ("LLM + validation", "fixed", True),
    ("Static + validated LLM", "static", True),
    ("Adaptive + validated LLM", "run_all", True),
]
COLORS = {"Static rules": "#eb6834", "Adaptive": "#2a78d6", "LLM + validation": "#e87ba4",
          "Static + validated LLM": "#1baf7a", "Adaptive + validated LLM": "#eda100"}
STANDARD = ["missing", "duplicate", "invalid_category", "outlier"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use rules from the offline Experiment 3 test")
    args, _ = ap.parse_known_args()
    sfx = "_mock" if args.mock else ""
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    scen = list(exp["semantic_scenarios"])
    llm = {ds: load_llm_rules(ds, args.mock) for ds in datasets}
    missing = [ds for ds in datasets if not llm[ds]]
    if missing:
        sys.exit(f"No Experiment 3 rules for {missing}. Run python -m experiments.exp3_llm_rules first.")

    rows = []
    for name, mode, uses_llm in CONFIGS:
        for ds, cfg in datasets.items():
            for r, extra in (llm[ds].items() if uses_llm else [(0, None)]):
                for sc in scen:
                    rows += [dict(x, configuration=name, run=r) for x in run(ds, sc, cfg, exp, mode, extra_rules=extra)]
    batch = pd.DataFrame(rows).fillna({c: 0 for c in rows[0] if c.startswith(("n_", "tp_"))})
    raw = results_dir("raw", f"exp3b_batch_results{sfx}.csv"); raw.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(raw, index=False)
    ev = batch[batch.evaluation == 1].fillna(0)

    def per_run(by):
        s = summarise(ev, by + ["run"])
        g = ev.groupby(by + ["run"])
        sem_n = g[["n_semantic_numeric", "n_semantic_category"]].sum().sum(axis=1)
        sem_tp = g[["tp_semantic_numeric", "tp_semantic_category"]].sum().sum(axis=1)
        std_n = g[[f"n_{t}" for t in STANDARD]].sum().sum(axis=1)
        std_tp = g[[f"tp_{t}" for t in STANDARD]].sum().sum(axis=1)
        extra = pd.DataFrame({
            "recall_semantic": sem_tp / sem_n,
            "recall_semantic_numeric": g["tp_semantic_numeric"].sum() / g["n_semantic_numeric"].sum(),
            "recall_semantic_category": g["tp_semantic_category"].sum() / g["n_semantic_category"].sum(),
            "recall_standard": std_tp / std_n}).reset_index()
        return s.merge(extra, on=by + ["run"])

    metrics = ["recall_semantic", "recall_semantic_numeric", "recall_semantic_category", "recall_standard",
               "precision", "recall", "f1", "fpr"]
    tables = results_dir("tables"); tables.mkdir(parents=True, exist_ok=True)
    order = {n: i for i, (n, _, _) in enumerate(CONFIGS)}

    def mean_table(by):
        p = per_run(by)
        m = p.groupby(by)[metrics].mean().reset_index()
        return m.sort_values(by, key=lambda s: s.map(order) if s.name == "configuration" else s)

    by_ds = mean_table(["dataset", "configuration"])
    by_ds.to_csv(tables / f"exp3b_semantic_detection{sfx}.csv", index=False)
    overall = mean_table(["configuration"])
    overall.to_csv(tables / f"exp3b_semantic_overall{sfx}.csv", index=False)
    by_sc = mean_table(["scenario", "configuration"])
    by_sc.to_csv(tables / f"exp3b_semantic_by_scenario{sfx}.csv", index=False)

    # which violations each configuration catches (recall per violated column)
    sem_cols = sorted({c[len("n_sem_"):] for c in ev.columns if c.startswith("n_sem_")})
    rec = []
    for (ds, cfgname), g in ev.groupby(["dataset", "configuration"]):
        what = {s["column"]: s["what"] for s in datasets[ds]["semantic_defects"]}
        for col in sem_cols:
            n = g[f"n_sem_{col}"].sum()
            if n:
                rec.append({"dataset": ds, "configuration": cfgname, "column": col, "violation": what.get(col),
                            "recall": g[f"tp_sem_{col}"].sum() / n})
    per_col = pd.DataFrame(rec).pivot_table(index=["dataset", "column", "violation"], columns="configuration",
                                            values="recall").reset_index()
    per_col = per_col[["dataset", "column", "violation"] + [n for n, _, _ in CONFIGS]]
    per_col.to_csv(tables / f"exp3b_semantic_by_violation{sfx}.csv", index=False)

    figs = results_dir("figures"); figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ds_list = list(datasets); x = np.arange(len(ds_list)); w = 0.8 / len(CONFIGS)
    for ax, (m, label) in zip(axes, [("recall_semantic", "Recall on semantic defects"),
                                     ("recall_standard", "Recall on standard defects"),
                                     ("fpr", "False-positive rate")]):
        for j, (n, _, _) in enumerate(CONFIGS):
            sub = by_ds[by_ds.configuration == n].set_index("dataset").reindex(ds_list)
            ax.bar(x + (j - (len(CONFIGS) - 1) / 2) * w, sub[m], w * 0.92, color=COLORS[n], label=n)
        ax.set_xticks(x, [DATASET_LABELS[d].replace(" Inspections", "") for d in ds_list], fontsize=8.5)
        ax.set_title(label, fontsize=10, loc="left", color="#1f1f1c")
        if m != "fpr":
            ax.set_ylim(0, 1.05)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower left", ncol=len(CONFIGS), bbox_to_anchor=(0.01, 0.0))
    fig.text(0.99, 0.065, "Scenarios S1 (stable) and S2 (drift), T6-T20. LLM configurations: mean over Experiment 3 runs."
             + (" MOCK LLM." if args.mock else ""), ha="right", fontsize=7.5, color="#55554f")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(figs / f"exp3b_semantic_defects{sfx}.png", dpi=300)
    plt.close(fig)

    summary = {r.configuration: {m: getattr(r, m) for m in metrics} for r in overall.itertuples()}
    log = write_run_log(f"exp3b_semantic_defects{sfx}", datasets, exp, batch, summary,
                        {"batch_results": str(raw), "tables": str(tables), "figures": str(figs)},
                        time.perf_counter() - t0)
    pd.set_option("display.width", 220)
    print(overall.round(4).to_string(index=False))
    print()
    print(by_ds.round(4).to_string(index=False))
    print()
    print(per_col.round(3).to_string(index=False))
    print(f"\nRun log: {log}")


if __name__ == "__main__":
    main()
