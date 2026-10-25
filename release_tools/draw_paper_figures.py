"""Redraw the three original manuscript data figures from saved v2 evidence.

Run with the project's scientific Python environment:
    python final_paper/tools/draw_figures.py

No experiment, model call, configuration edit, or original-result write occurs.
Each figure is saved as vector PDF/SVG and a 600-dpi PNG, then its exact plotting
data and source hashes are written under final_paper/qa/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
PAPER = WORKSPACE / "paper" / "latex"
PROJECT = WORKSPACE / "adaptive-data-quality"
RESULTS = PROJECT / "results" / "v2"
FIGURES = PAPER / "figures"
QA = WORKSPACE / ".artifacts" / "publication-figures"
FIGURES.mkdir(exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

# Okabe-Ito hues; identities also have different markers and line styles.
BLUE, TEAL, AMBER = "#0072B2", "#009E73", "#D55E00"
INK, MUTED, GRID, RULE = "#26323D", "#59636D", "#DDE2E6", "#9DA6AF"
DATASETS = [
    ("bts_airline", "BTS Airline", 6, 6),
    ("chicago_food", "Chicago Food Inspections", 4, 4),
    ("online_retail", "Online Retail II", 2, 2),
]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.65,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
})
VALIDATION = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axis_style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6, pad=3)


def save(fig, name):
    # Fixed physical dimensions make the type scale predictable at manuscript width.
    # All content is positioned within the figure instead of cropped differently
    # by each output backend.
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print(f"Saved {name}: PDF, SVG and 600-dpi PNG")


def h1a():
    raw_dir = RESULTS / "rq1_thresholds" / "test" / "raw"
    files = sorted(raw_dir.glob("*__seed*.parquet"))
    assert len(files) == 30, f"Expected 3 datasets x 10 seeds, found {len(files)} files"
    columns = ["dataset", "scenario", "method", "seed", "t", "drift_level", "evaluation", "FP", "N"]
    raw = pd.concat([pd.read_parquet(p, columns=columns) for p in files], ignore_index=True)
    methods = [
        ("adaptive", "Adaptive", BLUE, "o", "-"),
        ("ewma", "EWMA", TEAL, "^", (0, (4, 2))),
        ("fixed_pooled", "Fixed pooled warm-up", AMBER, "s", (0, (1.3, 1.6))),
    ]
    d = raw[
        (raw["scenario"] == "B_drift") & raw["evaluation"].astype(bool)
        & raw["method"].isin([x[0] for x in methods])
    ].copy()
    assert set(d["seed"]) == set(range(1001, 1011))
    assert set(d["t"]) == set(range(6, 21))
    g = d.groupby(["dataset", "method", "t"], as_index=False).agg(
        FP=("FP", "sum"), N=("N", "sum"), drift_level=("drift_level", "first"),
        seeds=("seed", "nunique"),
    )
    assert (g["seeds"] == 10).all()
    assert len(g) == 3 * 3 * 15
    g["fpr_pct"] = 100.0 * g["FP"] / g["N"]
    assert (g["N"] > 0).all()
    g.to_csv(QA / "h1a_plot_data.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
    for index, (ax, (ds, title, _, _)) in enumerate(zip(axes, DATASETS)):
        # Draw the proposed line last so it remains identifiable at coincidence.
        for method, label, color, marker, linestyle in reversed(methods):
            s = g[(g["dataset"] == ds) & (g["method"] == method)].sort_values("t")
            ax.plot(
                100 * s["drift_level"], s["fpr_pct"], label=label, color=color,
                linestyle=linestyle, linewidth=1.4, marker=marker,
                markersize=3.2, markerfacecolor="white", markeredgewidth=0.8,
                markevery=2, zorder=3 if method == "adaptive" else 2,
            )
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", pad=8)
        ax.set_xlim(0, 32)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Injected drift (%)", labelpad=5)
        if ds == "bts_airline":
            ax.set_ylim(0, 8)
            ax.set_yticks([0, 2, 4, 6, 8])
        elif ds == "chicago_food":
            ax.set_ylim(0, 0.30)
            ax.set_yticks([0, 0.10, 0.20, 0.30])
        else:
            ax.set_ylim(0, 0.90)
            ax.set_yticks([0, 0.30, 0.60, 0.90])
        axis_style(ax)
    axes[0].set_ylabel("False-positive rate (%)", labelpad=5)
    chicago = g[g["dataset"] == "chicago_food"].pivot(index="t", columns="method", values="fpr_pct")
    coincide = bool(np.allclose(chicago["adaptive"], chicago["ewma"], atol=1e-12, rtol=0))
    assert coincide
    axes[1].text(
        0.5, 0.045, "Adaptive and EWMA coincide",
        transform=axes[1].transAxes, ha="center", va="bottom", fontsize=6.6, color=MUTED,
    )
    handles = [
        Line2D([0], [0], color=c, marker=mk, linestyle=ls, linewidth=1.4,
               markerfacecolor="white", markersize=4, markeredgewidth=0.8, label=lab)
        for _, lab, c, mk, ls in methods
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.51, 0.995), columnspacing=2.0, handlelength=2.6)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.21, top=0.78, wspace=0.43)
    save(fig, "h1a_fpr_over_time")
    VALIDATION["h1a_fpr_over_time"] = {
        "data_rows": len(g), "source_files": len(files), "seeds": 10, "evaluation_batches_per_seed": 15,
        "aggregation": "100 * sum(FP) / sum(N), by dataset, method and evaluation batch",
        "scenario": "B_drift", "chicago_adaptive_ewma_coincide": coincide,
        "sources": {str(p.relative_to(WORKSPACE)): digest(p) for p in files},
    }


def guard():
    source = RESULTS / "rq2_category_guard" / "test" / "rq2b_summary.csv"
    values_source = RESULTS / "rq2_category_guard" / "test" / "rq2b_acceptance_per_value.csv"
    d = pd.read_csv(source)
    policies = [
        ("frozen", None, "Frozen warm-up categories"),
        ("recurrence", None, "Recurrence"),
        ("recurrence_manual", None, "Recurrence + dictionary-only rules"),
        ("recurrence_llm", "primary", "Recurrence + LLM rules (primary)"),
        ("recurrence_llm", "secondary", "Recurrence + LLM rules (secondary)"),
        ("recurrence_llm_value", "primary", "Recurrence + per-value LLM (primary)"),
        ("recurrence_llm_value", "secondary", "Recurrence + per-value LLM (secondary)"),
    ]
    selected = []
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.35), sharey=True)
    for index, (ax, (ds, title, n_valid, n_invalid)) in enumerate(zip(axes, DATASETS)):
        for row, (policy, model, _) in enumerate(policies):
            s = d[(d["dataset"] == ds) & (d["policy"] == policy)]
            s = s[s["model"].isna()] if model is None else s[s["model"] == model]
            assert set(s["kind"]) == {"valid", "invalid"} and len(s) == 2
            s = s.set_index("kind")
            v, iv = float(s.loc["valid", "accepted_share"]), float(s.loc["invalid", "accepted_share"])
            assert 0 <= v <= 1 and 0 <= iv <= 1
            selected.extend([
                {"dataset": ds, "policy": policy, "model": model or "", "kind": kind,
                 "accepted_share": float(s.loc[kind, "accepted_share"])}
                for kind in ["valid", "invalid"]
            ])
            ax.plot([100 * iv, 100 * v], [row, row], color=GRID, linewidth=1.3, zorder=1)
            # Fixed opposite offsets preserve visibility even when both values
            # are exactly zero or one; offsets encode no quantitative quantity.
            ax.scatter(100 * v, row - 0.105, s=24, marker="o", color=BLUE,
                       edgecolors="white", linewidth=0.5, zorder=3, clip_on=False)
            ax.scatter(100 * iv, row + 0.105, s=24, marker="s", color=AMBER,
                       edgecolors="white", linewidth=0.5, zorder=3, clip_on=False)
        ax.set_xlim(-5, 105)
        ax.set_xticks([0, 50, 100])
        short_title = {"bts_airline": "BTS Airline", "chicago_food": "Chicago Food",
                       "online_retail": "Online Retail II"}[ds]
        ax.set_title(f"({chr(97 + index)}) {short_title}\n{n_valid} valid / {n_invalid} invalid",
                     loc="left", fontsize=7.5, pad=9, linespacing=1.5)
        ax.set_ylim(len(policies) - 0.5, -0.5)
        axis_style(ax, grid="x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(policies)), [x[2] for x in policies], fontsize=7.0)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    handles = [
        Line2D([], [], color=BLUE, marker="o", linestyle="none", markersize=5,
               label="Valid new values"),
        Line2D([], [], color=AMBER, marker="s", linestyle="none", markersize=5,
               label="Invalid new values"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.65, 1.0), columnspacing=2.0)
    fig.text(0.6775, 0.025, "Novel category values admitted (%)",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.365, right=0.99, bottom=0.16, top=0.77, wspace=0.25)
    save(fig, "h2b_category_guard")
    plotting = pd.DataFrame(selected)
    plotting.to_csv(QA / "h2b_plot_data.csv", index=False)
    pvalues = pd.read_csv(values_source)
    distinct = pvalues.groupby(["dataset", "kind"])["value"].nunique()
    assert int(distinct.sum()) == 24
    for ds, _, valid, invalid in DATASETS:
        assert distinct.loc[(ds, "valid")] == valid
        assert distinct.loc[(ds, "invalid")] == invalid
    VALIDATION["h2b_category_guard"] = {
        "data_rows": len(plotting), "policy_model_rows": len(policies), "distinct_cases": int(distinct.sum()),
        "aggregation": "accepted_share copied exactly from summary; same seven policy/model rows as manuscript table",
        "offsets": "vertical +/-0.105 for visibility only; horizontal position encodes admission",
        "sources": {str(p.relative_to(WORKSPACE)): digest(p) for p in [source, values_source]},
    }


def temporal():
    source = RESULTS / "temporal_chicago" / "temporal_monthly.csv"
    raw = pd.read_csv(source)
    d = raw[
        (raw["drift_reference"] == "rolling")
        & ~raw["min_effect"].astype(bool) & ~raw["warmup"].astype(bool)
    ].copy().sort_values("month")
    assert len(d) == 196 and d["month"].is_unique
    assert d["month"].iloc[0] == "2010-06" and d["month"].iloc[-1] == "2026-09"
    assert int(d["records"].sum()) == 309502
    counts = d["label"].value_counts().to_dict()
    assert counts == {"drift": 143, "stable": 43, "drift+degradation": 10}, counts
    assert np.allclose(d["flagged_share"], d["flagged"] / d["records"])
    d["date"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
    d["date_end"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp(how="end")
    d["drift_indicated"] = d["label"].str.contains("drift")
    d["degradation_indicated"] = d["label"].str.contains("degradation")
    d.to_csv(QA / "temporal_plot_data.csv", index=False)
    fig, (ax, track) = plt.subplots(
        2, 1, figsize=(6.5, 3.25), sharex=True,
        gridspec_kw={"height_ratios": [4.3, 0.55], "hspace": 0.15},
    )
    ax.plot(d["date"], 100 * d["flagged_share"], color=BLUE, linewidth=1.05, zorder=2)
    degraded = d[d["degradation_indicated"]]
    ax.scatter(
        degraded["date"], 100 * degraded["flagged_share"], s=25, marker="D",
        facecolors=AMBER, edgecolors="white", linewidth=0.55, zorder=4,
    )
    ax.set_ylabel("Records flagged (%)", labelpad=6)
    ax.set_ylim(0, 18)
    ax.set_yticks([0, 5, 10, 15])
    axis_style(ax)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.spines["bottom"].set_visible(False)
    cp = pd.Timestamp("2018-07-01")
    for target in [ax, track]:
        target.axvline(cp, color=MUTED, linewidth=0.9, linestyle=(0, (3.5, 2.5)), zorder=5)
    ax.annotate(
        "July 2018\nprocedure change", xy=(cp, 16.9), xytext=(5, 0),
        textcoords="offset points", ha="left", va="top", fontsize=7.3, color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.9},
    )
    starts = mdates.date2num(d["date"])
    ends = mdates.date2num(d["date_end"])
    centers, widths = (starts + ends) / 2, ends - starts
    track.bar(centers, np.ones(len(d)), width=widths, color="#F1F3F5",
              linewidth=0, align="center", zorder=1)
    indicated = d["drift_indicated"].to_numpy()
    track.bar(centers[indicated], np.ones(indicated.sum()), width=widths[indicated],
              color=BLUE, linewidth=0, align="center", zorder=2)
    track.set_ylim(0, 1)
    track.set_yticks([0.5], ["Drift"])
    track.tick_params(axis="y", length=0, pad=8)
    track.spines[["left", "right", "top"]].set_visible(False)
    track.spines["bottom"].set_color(RULE)
    track.set_xticks(pd.to_datetime([f"{year}-01-01" for year in [2011, 2014, 2017, 2020, 2023, 2026]]))
    track.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    track.set_xlabel("Calendar year", labelpad=5)
    track.set_xlim(pd.Timestamp("2010-04-15"), pd.Timestamp("2026-11-15"))
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.3, label="Records flagged"),
        Line2D([], [], color=AMBER, marker="D", linestyle="none", markersize=4.5,
               label="Degradation indicated"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.54, 0.99), columnspacing=2)
    fig.text(0.99, 0.02, "Drift strip: blue = indicated; light grey = not indicated",
             ha="right", va="bottom", fontsize=6.8, color=MUTED)
    fig.subplots_adjust(left=0.095, right=0.99, top=0.84, bottom=0.19)
    save(fig, "chicago_temporal")
    VALIDATION["chicago_temporal"] = {
        "months": len(d), "first": str(d["month"].iloc[0]), "last": str(d["month"].iloc[-1]),
        "record_count": int(d["records"].sum()), "label_counts": counts,
        "reference": "rolling", "minimum_effect_floor": False,
        "known_event_marker": "2018-07-01",
        "source": {str(source.relative_to(WORKSPACE)): digest(source)},
    }


if __name__ == "__main__":
    # Deliberate one-by-one generation in original manuscript figure order.
    h1a()
    guard()
    temporal()
    (QA / "figure_data_validation.json").write_text(
        json.dumps(VALIDATION, indent=2) + "\n", encoding="utf-8"
    )
    print("All figure data assertions passed; original results remain read-only.")
