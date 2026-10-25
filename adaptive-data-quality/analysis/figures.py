"""Figures for the paper from the v2 result tables (the v2 runners write tables only; nothing drew them).

Every figure reads the tables of one split (default: the confirmatory test run) and is written as PNG (300 dpi)
and PDF to results/figures/v2/<split>/, then copied to ../paper/figures/v2/ when the paper folder exists. A figure
whose input table is missing is skipped with a note (e.g. the sensitivity figure before the sensitivity analysis
has run); the others are still drawn. Nothing here changes a result.

    python -m analysis.figures                   # test split
    python -m analysis.figures --split dev
    python -m analysis.figures --only h1a_fpr_over_time h3_tradeoff

Colours: categorical slots in fixed order (blue, orange, aqua; validated palette, at most three series per
panel), one-hue blue ramp for magnitudes, grey for reference marks; identity is always also given by a legend or
a direct label and a marker shape, never by colour alone.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import REPO_ROOT, v2_results_dir  # noqa: E402

# ----------------------------------------------------------------------------- style
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5df", "#ffffff"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated adjacent and all-pairs)
REF = "#9a9a93"                                        # reference marks (targets, zero lines)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GOOD, BAD = "#1f8a4c", "#c8352b"                        # status (holds / fails), always with a text mark
DS = {"bts_airline": "BTS Airline", "chicago_food": "Chicago Food Inspections", "online_retail": "Online Retail II"}
DS_ORDER = list(DS)
SCEN = {"A_stable": "A stable", "B_drift": "B drift", "C_degradation": "C degradation",
        "D_drift_degradation": "D drift + degr.", "E1_mild_stable": "E1 mild errors", "E2_mild_drift": "E2 mild + drift",
        "F1_contam_stable": "F1 contamination", "F2_contam_drift": "F2 contam. + drift", "Z_zero_defects": "Z zero defects",
        "H1_sparse_duplicates": "H1 sparse duplicates", "H2_sparse_category": "H2 sparse category",
        "H3_short_incident": "H3 short incident", "H4_recovery": "H4 recovery"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": REF, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "regular", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42})


def style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(REF)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6)


def note(fig, text):
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=6.5, color=MUTED)


class Figs:
    def __init__(self, split: str, out: Path, paper: Path | None):
        self.split, self.out, self.paper = split, out, paper
        self.done, self.skipped = [], []

    def table(self, exp: str, name: str, split: bool = True) -> pd.DataFrame:
        p = (v2_results_dir(exp, self.split) if split else v2_results_dir(exp)) / name
        if not p.exists():
            raise FileNotFoundError(str(p.relative_to(REPO_ROOT)))
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    def save(self, fig, name: str):
        self.out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        if self.paper is not None:
            self.paper.mkdir(parents=True, exist_ok=True)
            for ext in ("png", "pdf"):
                shutil.copy2(self.out / f"{name}.{ext}", self.paper / f"{name}.{ext}")
        self.done.append(name)


# ----------------------------------------------------------------------------- H1a
def h1a_fpr_over_time(F: Figs):
    """FPR per evaluation batch under gradual drift: adaptive vs the primary comparator vs the best rolling one."""
    raw_dir = v2_results_dir("rq1_thresholds", F.split) / "raw"
    files = sorted(raw_dir.glob("*__seed*.parquet"))
    if not files:
        raise FileNotFoundError(str((raw_dir / "*.parquet").relative_to(REPO_ROOT)))
    cols = ["dataset", "scenario", "method", "seed", "t", "drift_level", "evaluation", "FP", "N"]
    raw = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    meths = [("fixed_pooled", "Fixed, pooled warm-up", S2, "s"), ("ewma", "EWMA (rolling)", S3, "^"),
             ("adaptive", "Adaptive (proposed)", S1, "o")]              # proposed drawn last, on top
    d = raw[(raw.scenario == "B_drift") & (raw.evaluation == 1) & raw.method.isin([m for m, *_ in meths])]
    g = d.groupby(["dataset", "method", "t"]).agg(FP=("FP", "sum"), N=("N", "sum"), drift=("drift_level", "first")).reset_index()
    g["fpr"] = 100 * g.FP / g.N
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=False)
    for ax, ds in zip(axes, DS_ORDER):
        for m, lab, c, mk in meths:
            s = g[(g.dataset == ds) & (g.method == m)].sort_values("t")
            ax.plot(s.drift * 100, s.fpr, color=c, marker=mk, markersize=3.2, linewidth=1.5, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
        a = g[(g.dataset == ds) & (g.method == "adaptive")].set_index("t").fpr
        e = g[(g.dataset == ds) & (g.method == "ewma")].set_index("t").fpr
        if np.allclose(a.to_numpy(), e.reindex(a.index).to_numpy()):
            ax.text(0.03, 0.04, "adaptive and EWMA coincide", transform=ax.transAxes, fontsize=6.5, color=INK2)
        ax.set_title(DS[ds])
        ax.set_xlabel("Drift level (%)")
        ax.set_ylim(bottom=0)
        style(ax)
    axes[0].set_ylabel("False-positive rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    note(fig, f"Scenario B (gradual drift), evaluation batches T6-T20, {d.seed.nunique()} seeds pooled ({F.split}).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    F.save(fig, "h1a_fpr_over_time")


def h1a_forest(F: Figs):
    """Paired FPR and recall differences (adaptive minus comparator) with 95% seed-bootstrap CIs."""
    ci = F.table("rq1_thresholds", "rq1a_paired_ci.csv")
    comps = [("fixed_pooled", "vs fixed, pooled (primary)", S2, "s"), ("ewma", "vs EWMA", S3, "^"),
             ("rolling_mad_cal", "vs rolling MAD, calibrated", S1, "o")]
    scen = ["A_stable", "B_drift", "C_degradation", "D_drift_degradation", "E1_mild_stable", "E2_mild_drift",
            "F1_contam_stable", "F2_contam_drift", "Z_zero_defects"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey=True)
    for col, ds in enumerate(DS_ORDER):
        for row, metric in enumerate(("fpr", "recall")):
            ax = axes[row, col]
            for k, (cm, lab, c, mk) in enumerate(comps):
                s = ci[(ci.dataset == ds) & (ci.comparator == cm) & (ci.metric == metric)].set_index("scenario").reindex(scen)
                y = np.arange(len(scen)) + (k - 1) * 0.24
                ax.errorbar(100 * s.estimate, y, xerr=[100 * (s.estimate - s.ci_low), 100 * (s.ci_high - s.estimate)],
                            fmt=mk, color=c, markersize=3.6, elinewidth=1, capsize=0, label=lab,
                            markeredgecolor=SURFACE, markeredgewidth=0.5)
            ax.axvline(0, color=REF, linewidth=0.8)
            if metric == "recall":
                ax.axvline(-1, color=REF, linewidth=0.8, linestyle=(0, (2, 2)))
            ax.set_yticks(range(len(scen)), [SCEN[s] for s in scen])
            style(ax, grid="x")
            if row == 0:
                ax.set_title(DS[ds])
            ax.set_xlabel("FPR difference (points)" if metric == "fpr" else "Recall difference (points)")
    axes[0, 0].invert_yaxis()                     # shared y: invert once
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    note(fig, "Adaptive minus comparator; 95% paired seed-bootstrap CI. Dashed: recall non-inferiority margin (-1 point). "
              "Left of 0 = fewer false alarms (top) or lower recall (bottom).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    F.save(fig, "h1a_paired_differences")


# ----------------------------------------------------------------------------- H1b
def h1b_macro_f1(F: Figs):
    t = F.table("rq1_classification", "rq1b_classification_summary.csv")
    ci = F.table("rq1_classification", "rq1b_paired_ci.csv")
    ci = ci[ci.metric == "macro_f1"]
    names = {"proposed": "Proposed (gated drift + adaptive)", "gated_ewma": "Gated drift + EWMA",
             "gated_fixed_pooled": "Gated drift + fixed", "gated_rolling_mad": "Gated drift + rolling MAD",
             "proposed_min_effect": "Proposed + practical drift floor", "proposed_no_gate": "Proposed without gate",
             "tfdv_warmup_calibrated": "TFDV, calibrated", "deequ_anomaly_calibrated": "Deequ anomaly, calibrated",
             "avh_style_quarantine": "AVH-style (quarantine)", "static_no_drift": "Static rules, no drift signal",
             "tfdv_warmup": "TFDV as documented", "deequ_anomaly": "Deequ as documented"}
    order = list(names)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)
    for ax, ds in zip(axes, DS_ORDER):
        s = t[t.dataset == ds].set_index("comparator").reindex(order)
        c = ci[ci.dataset == ds].set_index("proposed_minus")
        y = np.arange(len(order))
        for i, comp in enumerate(order):
            v = s.loc[comp, "macro_f1"]
            if pd.isna(v):
                continue
            is_p = comp == "proposed"
            ax.plot([0, v], [i, i], color=GRID, linewidth=1, zorder=1)
            ax.plot(v, i, "o", color=S1 if is_p else INK2, markersize=4.5 if is_p else 3.5, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            if not is_p and comp in c.index:   # proposed minus comparator CI excludes 0 -> mark
                r = c.loc[comp]
                tag = "" if r.ci_low > 0 else ("=" if r.ci_high >= 0 else "+")
                if tag:
                    ax.text(v + 0.02, i, "n.s." if tag == "=" else "better", va="center", fontsize=6, color=INK2)
        ax.axvline(s.loc["proposed", "macro_f1"], color=S1, linewidth=0.7, alpha=0.5, zorder=0)
        ax.set_yticks(y, [names[o] for o in order])
        ax.set_xlim(0, 1.08)
        ax.set_title(DS[ds])
        ax.set_xlabel("Four-class macro-F1")
        style(ax, grid="x")
    axes[0].invert_yaxis()
    note(fig, "Batch labels stable / drift / degradation / both, scenarios A-D, all seeds. 'n.s.': proposed not "
              "significantly better (paired 95% CI includes 0); 'better': comparator significantly better.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    F.save(fig, "h1b_macro_f1")


def h1b_confusion(F: Figs):
    cm = F.table("rq1_classification", "rq1b_confusion_matrices.csv")
    cm = cm[cm.comparator == "proposed"]
    classes = ["stable", "drift", "degradation", "drift+degradation"]
    lab = ["stable", "drift", "degr.", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7fafe"] + BLUE_RAMP)
    for ax, ds in zip(axes, DS_ORDER):
        m = cm[cm.dataset == ds].set_index("row_0").reindex(classes)[classes].to_numpy(dtype=float)
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                        color=SURFACE if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), lab)
        ax.set_yticks(range(4), lab if ds == DS_ORDER[0] else [""] * 4)
        ax.set_xlabel("Predicted")
        if ds == DS_ORDER[0]:
            ax.set_ylabel("True")
        ax.set_title(DS[ds])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    note(fig, "Proposed classifier, evaluation batches of scenarios A-D, all seeds; shade = share of the true class.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    F.save(fig, "h1b_confusion")


# ----------------------------------------------------------------------------- H2
def h2a_semantic(F: Figs):
    t = F.table("rq2_coverage", "rq2a_summary.csv")
    t = t[(t.scenario == "S2_semantic_drift") & ((t.model.isna()) | (t.model == "primary")) &
          ((t.redundancy.isna()) | (t.redundancy == "logical"))]
    configs = [("stat_fixed", "Statistical, fixed"), ("stat_adaptive", "Statistical, adaptive"),
               ("manual", "Manual rules (admitted)"), ("llm_validated", "LLM rules (validated)"),
               ("fixed+llm_validated", "Fixed + LLM"), ("adaptive+llm_validated", "Adaptive + LLM")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, ds in zip(axes, DS_ORDER):
        s = t[t.dataset == ds].set_index("config")
        for i, (c, lab) in enumerate(configs):
            v = s.loc[c, "semantic_recall_mean"] if c in s.index else np.nan
            col = S1 if "llm" in c else INK2
            ax.barh(i, v, height=0.55, color=col if "llm" in c else "#b9b8b1")
            if pd.notna(v):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(configs)), [l for _, l in configs])
        if ax is axes[0]:                       # shared y: invert once
            ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_title(DS[ds])
        ax.set_xlabel("Semantic-defect recall")
        style(ax, grid="x")
    note(fig, "Scenario S2 (semantic defects under drift), primary LLM, mean over 5 generations and all seeds. "
              "Blue: rule sets with LLM rules.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    F.save(fig, "h2a_semantic_recall")


def h2b_guard(F: Figs):
    t = F.table("rq2_category_guard", "rq2b_summary.csv")
    pol = [("frozen", "Frozen (warm-up only)"), ("recurrence", "Recurrence (accept after 2 batches)"),
           ("recurrence_manual", "Recurrence + manual rules"), ("recurrence_llm", "Recurrence + LLM rules"),
           ("recurrence_llm_value", "Recurrence + per-value LLM check")]
    t = t[(t.model.isna()) | (t.model == "primary")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, ds in zip(axes, DS_ORDER):
        s = t[t.dataset == ds]
        for i, (p, _) in enumerate(pol):
            g = s[s.policy == p].set_index("kind")
            if not len(g):
                continue
            v, iv = g.loc["valid", "accepted_share"], g.loc["invalid", "accepted_share"]
            ax.plot([iv, v], [i, i], color=GRID, linewidth=2, zorder=1)
            off = 0.13 if abs(v - iv) < 0.04 else 0.0   # identical shares: offset so both marks stay visible
            ax.plot(v, i - off, "o", color=S1, markersize=5, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    label="valid new values" if i == 0 else None)
            ax.plot(iv, i + off, "s", color=S2, markersize=5, zorder=3, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    label="invalid new values" if i == 0 else None)
        ax.set_yticks(range(len(pol)), [l for _, l in pol])
        if ax is axes[0]:                       # shared y: invert once
            ax.invert_yaxis()
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(DS[ds])
        ax.set_xlabel("Share accepted as valid")
        style(ax, grid="x")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.02))
    try:
        pv = F.table("rq2_category_guard", "rq2b_acceptance_per_value.csv")
        n = pv.groupby(["dataset", "kind"])["value"].nunique()
        txt = "; ".join(f"{DS[d].split()[0]} {n[(d, 'valid')]} valid + {n[(d, 'invalid')]} invalid" for d in DS_ORDER
                        if (d, "valid") in n)
        txt = f" Distinct new values (the same in every seed): {txt}."
    except FileNotFoundError:
        txt = ""
    note(fig, "Ideal: valid near 1, invalid near 0." + txt)
    fig.subplots_adjust(left=0.27, right=0.99, bottom=0.24, top=0.82, wspace=0.12)
    F.save(fig, "h2b_category_guard")


# ----------------------------------------------------------------------------- H3
def h3_tradeoff(F: Figs):
    d = F.table("rq3_scheduling", "rq3_h3_decisions.csv")
    d = d[d.post_onset_recall.notna() & (d.policy != "all")]
    pol = [("risk", "Risk (proposed)"), ("periodic", "Periodic*"), ("random", "Random*"),
           ("failure_only", "Failure only"), ("drift_only", "Drift only")]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.6), sharey=True, sharex=True)
    for ax, (p, lab) in zip(axes, pol):
        s = d[d.policy == p]
        for ds, mk, c in zip(DS_ORDER, ("o", "s", "^"), (S1, S2, S3)):
            q = s[s.dataset == ds]
            ax.scatter(q.saving_checks_pct, q.post_onset_recall, s=14, marker=mk, color=c, edgecolors=SURFACE,
                       linewidths=0.5, zorder=3, label=DS[ds])
        ax.axhline(0.90, color=REF, linewidth=0.8, linestyle=(0, (2, 2)))
        ax.set_title(lab, fontsize=7.5)
        ax.set_xlabel("Checks saved (%)")
        style(ax)
        ax.text(0.98, 0.04, f"useful {int(s.useful.sum())}/{len(s)}", transform=ax.transAxes, ha="right", fontsize=6.5,
                color=INK2)
    axes[0].set_ylabel("Post-onset recall")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    note(fig, "Degradation and incident scenarios (C, D, H1-H4). Dashed: detection target 0.90. "
              "Useful = saves checks and meets the target. *Cost-matched to risk.")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.80, wspace=0.12)
    F.save(fig, "h3_cost_detection")


def h3_runtime(F: Figs):
    r = F.table("rq3_scheduling", "rq3_runtime.csv")
    r = r[(r.scenario == "A_stable") & r.policy.isin(["all", "risk"])]
    groups = [("Profile", ["ms_profile_median"]), ("Drift", ["ms_drift_median"]), ("Execute checks", ["ms_execute_median"]),
              ("Other", ["ms_read_median", "ms_adapt_median", "ms_select_median", "ms_classify_median", "ms_persist_median"])]
    cols = ["#256abf", "#6da7ec", S2, "#b9b8b1"]
    rows = []
    for ds in DS_ORDER:
        for pol in ("all", "risk"):
            for reuse in (False, True):
                q = r[(r.dataset == ds) & (r.policy == pol) & (r.reuse.astype(str) == str(reuse))]
                if len(q):
                    rows.append((ds, pol, reuse, [float(q[c].sum(axis=1).iloc[0]) for _, c in groups]))
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ylab = []
    for i, (ds, pol, reuse, vals) in enumerate(rows):
        left = 0
        for (g, _), c, v in zip(groups, cols, vals):
            ax.barh(i, v, left=left, height=0.62, color=c, edgecolor=SURFACE, linewidth=1, label=g if i == 0 else None)
            left += v
        ax.text(left + 8, i, f"{left:.0f} ms", va="center", fontsize=6.5, color=INK2)
        ylab.append(f"{DS[ds].split()[0]}  {'run-all' if pol == 'all' else 'risk'}{', reuse' if reuse else ''}")
    ax.set_yticks(range(len(rows)), ylab)
    ax.invert_yaxis()
    ax.set_xlabel("Median monitoring time per batch (ms)")
    style(ax, grid="x")
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    note(fig, "Scenario A, timed trials (5 repeats, first test seed). 'reuse': checks reuse the batch profile's statistics.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    F.save(fig, "h3_runtime_phases")


# ----------------------------------------------------------------------------- ablation, external, temporal
def ablation_effects(F: Figs):
    e = F.table("ablation", "ablation_main_effects.csv")
    metrics = [("f1", "Record-level F1"), ("semantic_recall", "Semantic recall"), ("macro_f1", "Batch-label macro-F1")]
    factors = [("adaptation", "Adaptive vs fixed thresholds"), ("llm", "Validated LLM rules vs none"),
               ("selection", "Risk selection vs run-all")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, (m, mlab) in zip(axes, metrics):
        for k, (ds, mk, c) in enumerate(zip(DS_ORDER, ("o", "s", "^"), (S1, S2, S3))):
            s = e[e.dataset == ds].set_index("factor")
            y = np.arange(len(factors)) + (k - 1) * 0.2
            ax.plot([s.loc[f, m] for f, _ in factors], y, mk, color=c, markersize=4.2, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=0.5, label=DS[ds])
        ax.axvline(0, color=REF, linewidth=0.8)
        ax.set_yticks(range(len(factors)), [l for _, l in factors])
        if ax is axes[0]:                       # shared y: invert once
            ax.invert_yaxis()
        ax.set_title(mlab)
        ax.set_xlabel("Main effect (difference)")
        style(ax, grid="x")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.6, 1.02))
    note(fig, "Eight-way ablation; effect = mean over the other factors' levels (scenarios A-D, S1, S2).")
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.25, top=0.78, wspace=0.15)
    F.save(fig, "ablation_main_effects")


def external_tax(F: Figs):
    t = F.table("external", "external_detection.csv")
    t = t[t.benchmark == "tax"]
    order = [("fixed_pooled", "Fixed, pooled"), ("adaptive", "Adaptive"), ("adaptive+llm_primary", "Adaptive + LLM rules"),
             ("adaptive+deps", "Adaptive + learned dependencies"), ("adaptive+xcol_primary", "Adaptive + LLM cross-column"),
             ("adaptive+llm_primary+xcol_primary", "Adaptive + LLM + LLM cross-column")]
    s = t.set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
    for ax, (m, lab) in zip(axes, (("cell_recall", "Cell recall"), ("fpr", "Record false-positive rate"))):
        for i, (k, _) in enumerate(order):
            v = s.loc[k, m] if k in s.index else np.nan
            ax.barh(i, v, height=0.55, color=S1 if "xcol" in k or "deps" in k else "#b9b8b1")
            ax.text(v + (0.015 if m == "cell_recall" else 0.0008), i, f"{v:.3f}" if m == "fpr" else f"{v:.2f}",
                    va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(order)), [l for _, l in order])
        if ax is axes[0]:                       # shared y: invert once
            ax.invert_yaxis()
        ax.set_title(lab)
        style(ax, grid="x")
        if m == "cell_recall":
            ax.set_xlim(0, 1.1)
        else:
            ax.set_xlim(0, s.loc[[k for k, _ in order if k in s.index], "fpr"].max() * 1.35)
    note(fig, "Tax dirty/clean benchmark, all seeds. Blue: with cross-column rules.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    F.save(fig, "external_tax")


def chicago_monthly(F: Figs):
    m = F.table("temporal_chicago", "temporal_monthly.csv", split=False)
    m = m[(m.drift_reference == "rolling") & (~m.min_effect.astype(bool)) & (~m.warmup.astype(bool))].copy()
    m["date"] = pd.PeriodIndex(m.month, freq="M").to_timestamp()
    lab = m.label.astype(str)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 2.9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(m.date, 100 * m.flagged_share, color=S1, linewidth=1.2)
    deg = m[lab.str.contains("degradation")]
    ax.plot(deg.date, 100 * deg.flagged_share, "o", color=S2, markersize=3.2, markeredgecolor=SURFACE,
            markeredgewidth=0.5, label="month labelled degradation", zorder=3)
    cp = pd.Timestamp("2018-07-01")
    for a in (ax, ax2):
        a.axvline(cp, color=REF, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(cp, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] else 1, " 1 Jul 2018 procedure change", fontsize=6.5,
            color=INK2, va="top")
    ax.set_ylabel("Records flagged (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    style(ax)
    drift = lab.str.contains("drift").astype(int)
    ax2.bar(m.date, drift, width=25, color="#6da7ec")
    ax2.set_yticks([0, 1], ["", "drift"])
    ax2.set_ylim(0, 1.2)
    style(ax2, grid=None)
    note(fig, "Raw City of Chicago food inspections replayed month by month through the frozen monitor (rolling drift "
              "reference, every check).")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    F.save(fig, "chicago_temporal")


def chicago_adjudication(F: Figs):
    adj = v2_results_dir("temporal_chicago") / "adjudication"
    p = adj / "facility_type_alarm_verdicts.csv"
    prov = False
    if not p.exists() or pd.read_csv(p).final_verdict.eq("pending").all():
        p, prov = adj / "facility_type_alarm_verdicts_provisional.csv", True
    if not p.exists():
        raise FileNotFoundError(str(p.relative_to(REPO_ROOT)))
    t = pd.read_csv(p)
    t["src"] = np.where(t.verdict_source.str.startswith("human"), "human",
                        np.where(t.verdict_source.str.startswith("rule"), "rule", "model/pending"))
    order = ["not_error", "unclear", "error", "pending"]
    srcs = [("human", "#256abf"), ("rule", "#6da7ec"), ("model/pending", "#cde2fb")]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for i, v in enumerate(order):
        left = 0
        for s, c in srcs:
            x = t[(t.final_verdict == v) & (t.src == s)].alarms.sum()
            if x:
                ax.barh(i, x, left=left, height=0.6, color=c, edgecolor=SURFACE, linewidth=1)
            left += x
        if left:
            ax.text(left + 15, i, f"{int(left)} ({100 * left / t.alarms.sum():.0f}%)", va="center", fontsize=6.5, color=INK2)
    for s, c in srcs:
        ax.barh(np.nan, 0, color=c, label={"human": "human verdict", "rule": "rule pre-label",
                                            "model/pending": "model suggestion"}[s])
    ax.set_yticks(range(len(order)), ["not an error", "unclear", "error", "pending"])
    ax.invert_yaxis()
    ax.set_xlabel("Facility Type alarms (evaluation months)")
    style(ax, grid="x")
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    note(fig, "PROVISIONAL: no human verdicts yet." if prov else "Verdicts from the value codebook.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    F.save(fig, "chicago_facility_type_adjudication")


def sensitivity_grid(F: Figs):
    c = F.table("sensitivity", "sens_conclusions.csv")
    c["row"] = np.where(c.dial == "default", "default", c.dial + ": " + c.setting.astype(str))
    claims = sorted(c.claim.unique())
    short = {k: k.split(" ")[0] for k in claims}
    cols = [(cl, ds) for cl in claims for ds in DS_ORDER]
    rows = list(dict.fromkeys(["default"] + [r for r in c.row if r != "default"]))
    piv = {(r.row, r.claim, r.dataset): r.holds for r in c.itertuples()}
    fig, ax = plt.subplots(figsize=(7.2, 0.16 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        for j, (cl, ds) in enumerate(cols):
            h = piv.get((r, cl, ds))
            if h is None:
                continue
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, color=GOOD if h else BAD, alpha=0.85 if h else 0.9,
                                       linewidth=0))
            ax.text(j, i, "✓" if h else "✗", ha="center", va="center", fontsize=5.5, color=SURFACE)
    ax.set_xlim(-0.6, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.set_xticks(range(len(cols)), [f"{short[cl]} {DS[ds].split()[0]}" for cl, ds in cols], rotation=90, fontsize=6)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    note(fig, "\n".join(f"{short[k]}: {k.split(' ', 1)[1]}" for k in claims))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    F.save(fig, "sensitivity_claims")


FIGURES = {"h1a_fpr_over_time": h1a_fpr_over_time, "h1a_paired_differences": h1a_forest, "h1b_macro_f1": h1b_macro_f1,
           "h1b_confusion": h1b_confusion, "h2a_semantic_recall": h2a_semantic, "h2b_category_guard": h2b_guard,
           "h3_cost_detection": h3_tradeoff, "h3_runtime_phases": h3_runtime, "ablation_main_effects": ablation_effects,
           "external_tax": external_tax, "chicago_temporal": chicago_monthly,
           "chicago_facility_type_adjudication": chicago_adjudication, "sensitivity_claims": sensitivity_grid}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--only", nargs="*", choices=list(FIGURES), default=list(FIGURES))
    ap.add_argument("--no-paper", action="store_true", help="do not copy into ../paper/figures/v2")
    args = ap.parse_args(argv)
    out = REPO_ROOT / "results" / "figures" / "v2" / args.split
    paper_root = REPO_ROOT.parent / "paper"
    paper = paper_root / "figures" / "v2" if (paper_root.exists() and not args.no_paper and args.split == "test") else None
    F = Figs(args.split, out, paper)
    for name in args.only:
        try:
            FIGURES[name](F)
        except FileNotFoundError as exc:
            F.skipped.append(f"{name}: input missing ({exc})")
        except Exception:  # noqa: BLE001
            F.skipped.append(f"{name}: FAILED\n{traceback.format_exc()}")
    print(f"[figures] {len(F.done)} written to {out}" + (f" and {paper}" if paper else ""))
    for n in F.done:
        print(f"  {n}.png / .pdf")
    for s in F.skipped:
        print(f"  skipped {s}")


if __name__ == "__main__":
    main()
