"""Section 14: Sensitivity Analysis.

Purpose            Show that the results "are not dependent on one manually selected parameter
                   setting" (project description, Section 14).
Design             One setting at a time: each dial below is changed while every other setting
                   stays at its default in configs/experiment.yaml. The default setting is also
                   repeated with five seeds, which shows how much results move by chance alone.

    Dial                 Values (configs/experiment.yaml: sensitivity)   Experiments re-run
    seed                 42 (default), 43, 44, 45, 46                    1, 2 and 4, all scenarios
    defect_rate          1 (default), 3, 5, 10, 20 % of records          1 and 4, scenarios A and B
    outlier_severity     outliers at 3-5, 4-12 (default), 5-8, 8-12 sd   1, scenarios A and B
    calibration_z        drift/degradation thresholds mean + z sd,       2, all scenarios
                         z = 2, 2.5, 3 (default), 3.5, 4
    risk_weights         each weight x0, x0.5, x2 (renormalised),        4, all scenarios
                         and equal weights
    risk_thresholds      moderate, high, very high, one at a time        4, all scenarios

Dials that change the data (seed, defect rate, severity) rebuild batches, ground truth and
profiles from data/clean in a separate folder (data/sensitivity_work), so the main experimental
data is never modified. The main data (python -m experiments.prepare_data) must exist first.

Conclusions        Each main claim is re-checked in every setting (sens_conclusions.csv):
                   C1 adaptive has a lower false-positive rate than static under drift
                   C2 adaptive recall stays within 0.02 of static recall under drift
                   C3 adaptive classifies batches more accurately than static (Experiment 2)
                   C4 selective execution scans less data than run-all in every scenario
                   C5 selective recall stays >= 0.90 once degradation starts (scenarios C, D)

Outputs            results/sensitivity/tables/*.csv, results/sensitivity/figures/*.png,
                   results/sensitivity/raw/ (cached per-setting results; a rerun skips settings
                   already done unless --force), results/raw/run_logs/sensitivity_*.json

    python -m experiments.sensitivity                       # every dial
    python -m experiments.sensitivity --dial seed defect_rate
    python -m experiments.sensitivity --force               # recompute cached settings
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (data_dir, datasets_config, experiment_config, now_iso, results_dir,  # noqa: E402
                          workspace, write_json)
from experiments.cost_aware import run as cost_run  # noqa: E402
from experiments.detection import run_all as detection_run_all  # noqa: E402
from experiments.exp2_drift_vs_degradation import classify, summarise_classification  # noqa: E402
from experiments.exp4_cost_optimization import summarise  # noqa: E402
from experiments.prepare_data import build_scenarios  # noqa: E402
from experiments.reporting import COLORS, DATASET_LABELS, LABELS, MARKERS, aggregate, style_axes  # noqa: E402

DIALS = ["seed", "defect_rate", "outlier_severity", "calibration_z", "risk_weights", "risk_thresholds"]
ALL_SCEN = ["A_stable", "B_drift", "C_degradation", "D_drift_degradation"]
AB = ["A_stable", "B_drift"]
OUT = results_dir("sensitivity")
RAW, TABLES, FIGS = OUT / "raw", OUT / "tables", OUT / "figures"
WORK = data_dir("sensitivity_work")
FORCE = False
KEEP = False


# ----------------------------------------------------------------------------- helpers
def cached(name: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Per-setting results are saved once; a rerun (e.g. after an interruption) reuses them."""
    path = RAW / f"{name}.csv"
    if path.exists() and not FORCE:
        return pd.read_csv(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def data_changed(exp: Dict[str, Any], base: Dict[str, Any]) -> bool:
    return any(exp[k] != base[k] for k in ("seed", "schedules", "defects"))


def with_data(name: str, exp: Dict[str, Any], base: Dict[str, Any], scenarios: List[str],
              datasets: Dict[str, Any], fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Run fn on the main data if the setting leaves the data unchanged; otherwise build the
    batches for this setting in a separate folder first (from data/clean), then remove them."""
    if not data_changed(exp, base):
        return fn()
    ws = WORK / name
    with workspace(ws):
        for ds, cfg in datasets.items():
            clean = pd.read_parquet(data_dir("clean", f"{ds}.parquet"))
            build_scenarios(ds, cfg, exp, clean, {s: exp["scenarios"][s] for s in scenarios}, verbose=False)
        out = fn()
    if not KEEP:
        shutil.rmtree(ws, ignore_errors=True)
    return out


def detection(exp: Dict[str, Any], datasets, scenarios: List[str]) -> pd.DataFrame:
    e = copy.deepcopy(exp)
    e["scenarios"] = {s: exp["scenarios"][s] for s in scenarios}
    return detection_run_all(datasets, e)


def cost(exp: Dict[str, Any], datasets, scenarios: List[str], modes=("run_all", "selective")) -> pd.DataFrame:
    rows = []
    for ds, cfg in datasets.items():
        for sc in scenarios:
            for m in modes:
                rows += cost_run(ds, sc, cfg, exp, m)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- metrics (long format)
def exp1_metrics(batch: pd.DataFrame) -> List[Dict[str, Any]]:
    """Static vs adaptive, evaluation batches, per dataset and overall, per scenario present."""
    ev = batch[batch.evaluation == 1]
    rows = []
    for scope_all in (False, True):
        by = ["scenario", "method"] if scope_all else ["dataset", "scenario", "method"]
        agg = aggregate(ev, by)
        for r in agg.to_dict("records"):
            scope = f"ALL|{r['scenario']}" if scope_all else f"{r['dataset']}|{r['scenario']}"
            for k in ("precision", "recall", "f1", "fpr"):
                rows.append({"experiment": "exp1", "scope": scope, "metric": f"{r['method']}_{k}", "value": r[k]})
        # outlier recall (defect severity dial)
        for keys, g in ev.groupby(by):
            keys = dict(zip(by, keys))
            scope = f"ALL|{keys['scenario']}" if scope_all else f"{keys['dataset']}|{keys['scenario']}"
            n = g["n_outlier"].sum()
            rows.append({"experiment": "exp1", "scope": scope, "metric": f"{keys['method']}_recall_outlier",
                         "value": g["tp_outlier"].sum() / n if n else None})
    return rows


def exp2_metrics(batch: pd.DataFrame, datasets, exp, z=None) -> List[Dict[str, Any]]:
    pred, _ = classify(batch, datasets, exp, z)
    _, acc = summarise_classification(pred, datasets)
    rows = []
    for r in acc.to_dict("records"):
        for k in ("accuracy_4class", "drift_detection_rate", "drift_false_alarm_rate", "degradation_detection_rate",
                  "degradation_false_alarm_rate", "false_degradation_alarms_during_pure_drift"):
            rows.append({"experiment": "exp2", "scope": r["dataset"], "metric": f"{r['method']}_{k}", "value": r[k]})
    return rows


def exp4_metrics(sel: pd.DataFrame, run_all: pd.DataFrame) -> List[Dict[str, Any]]:
    """Selective vs run-all (both adaptive), evaluation batches, per scenario and overall."""
    sel, ra = sel[sel.evaluation == 1], run_all[run_all.evaluation == 1]
    rows = []
    groups = [(f"ALL|{sc}", sel[sel.scenario == sc], ra[ra.scenario == sc]) for sc in sorted(sel.scenario.unique())]
    groups.append(("ALL|ALL", sel, ra))
    for scope, s, r in groups:
        ss, rr = summarise(s.assign(k=1), ["k"]).iloc[0], summarise(r.assign(k=1), ["k"]).iloc[0]
        rows += [{"experiment": "exp4", "scope": scope, "metric": "selective_recall", "value": ss.recall},
                 {"experiment": "exp4", "scope": scope, "metric": "run_all_recall", "value": rr.recall},
                 {"experiment": "exp4", "scope": scope, "metric": "selective_precision", "value": ss.precision},
                 {"experiment": "exp4", "scope": scope, "metric": "saving_data_pct",
                  "value": 100 * (1 - ss.bytes_scanned / rr.bytes_scanned)},
                 {"experiment": "exp4", "scope": scope, "metric": "saving_rules_pct",
                  "value": 100 * (1 - ss.rules_executed / rr.rules_executed)}]
    return rows


def tag(rows: List[Dict[str, Any]], dial: str, setting: str, value: Any) -> List[Dict[str, Any]]:
    return [{"dial": dial, "setting": setting, "value_setting": value, **r} for r in rows]


# ----------------------------------------------------------------------------- dials
def dial_seed(exp, datasets, sens) -> List[Dict[str, Any]]:
    out = []
    for seed in sens["seeds"]:
        e = copy.deepcopy(exp); e["seed"] = seed
        name = f"seed_{seed}"
        print(f"  seed {seed}")

        def work():
            det = detection(e, datasets, ALL_SCEN)
            cst = cost(e, datasets, ALL_SCEN)
            classes = pd.DataFrame(exp2_metrics(det, datasets, e))
            return pd.concat([det.assign(_kind="det"), cst.assign(_kind="cost"),
                              classes.assign(_kind="exp2")], ignore_index=True)
        # the default seed is the main data; its results are shared with the other dials
        if seed == exp["seed"]:
            det, cst = main_detection(exp, datasets), main_cost(exp, datasets)
            rows = exp1_metrics(det) + exp2_metrics(det, datasets, exp) + \
                exp4_metrics(cst[cst["mode"] == "selective"], cst[cst["mode"] == "run_all"])
        else:
            df = cached(name, lambda: with_data(name, e, exp, ALL_SCEN, datasets, work))
            det, cst = df[df._kind == "det"], df[df._kind == "cost"]
            rows = exp1_metrics(det) + df[df._kind == "exp2"][["experiment", "scope", "metric", "value"]].to_dict("records") + \
                exp4_metrics(cst[cst["mode"] == "selective"], cst[cst["mode"] == "run_all"])
        out += tag(rows, "seed", str(seed), seed)
    return out


def dial_defect_rate(exp, datasets, sens) -> List[Dict[str, Any]]:
    out = []
    for rate in sens["defect_rates"]:
        e = copy.deepcopy(exp)
        e["schedules"]["defects_baseline"] = [rate] * exp["n_batches"]
        name = f"defect_rate_{rate:g}"
        print(f"  defect rate {rate:.0%}")

        def work():
            det = detection(e, datasets, AB)
            cst = cost(e, datasets, AB)
            return pd.concat([det.assign(_kind="det"), cst.assign(_kind="cost")], ignore_index=True)
        df = cached(name, lambda: with_data(name, e, exp, AB, datasets, work))
        det, cst = df[df._kind == "det"], df[df._kind == "cost"]
        rows = exp1_metrics(det) + exp4_metrics(cst[cst["mode"] == "selective"], cst[cst["mode"] == "run_all"])
        out += tag(rows, "defect_rate", f"{rate:.0%}", rate)
    return out


def dial_outlier_severity(exp, datasets, sens) -> List[Dict[str, Any]]:
    out = []
    for label, sig in sens["outlier_sigma"].items():
        e = copy.deepcopy(exp)
        e["defects"]["outlier_sigma"] = list(sig)
        name = f"outlier_{sig[0]}_{sig[1]}"
        print(f"  outliers {label}")
        det = cached(name, lambda: with_data(name, e, exp, AB, datasets, lambda: detection(e, datasets, AB)))
        out += tag(exp1_metrics(det), "outlier_severity", label, f"{sig[0]}-{sig[1]}")
    return out


def dial_calibration_z(exp, datasets, sens) -> List[Dict[str, Any]]:
    det = main_detection(exp, datasets)
    out = []
    for z in sens["calibration_z"]:
        print(f"  z = {z}")
        out += tag(exp2_metrics(det, datasets, exp, z), "calibration_z", f"z = {z:g}", z)
    return out


def weight_variants(w: Dict[str, float], factors) -> Dict[str, Dict[str, float]]:
    v = {"plan (Section 11)": dict(w), "equal (0.2 each)": {k: 1 / len(w) for k in w}}
    for comp in w:
        for f in factors:
            raw = {k: (x * f if k == comp else x) for k, x in w.items()}
            tot = sum(raw.values())
            label = f"{comp} removed" if f == 0 else f"{comp} x{f:g}"
            v[label] = {k: x / tot for k, x in raw.items()}
    return v


def dial_risk_weights(exp, datasets, sens) -> List[Dict[str, Any]]:
    ra = main_cost(exp, datasets)
    ra = ra[ra["mode"] == "run_all"]
    out = []
    for label, w in weight_variants(exp["risk"]["weights"], sens["weight_factors"]).items():
        e = copy.deepcopy(exp); e["risk"]["weights"] = w
        name = "weights_" + "_".join(f"{k}{v:.3f}" for k, v in w.items())
        print(f"  weights {label}")
        sel = cached(name, lambda: cost(e, datasets, ALL_SCEN, modes=("selective",)))
        out += tag(exp4_metrics(sel, ra), "risk_weights", label,
                   " ".join(f"{k}={v:.2f}" for k, v in w.items()))
    return out


def dial_risk_thresholds(exp, datasets, sens) -> List[Dict[str, Any]]:
    ra = main_cost(exp, datasets)
    ra = ra[ra["mode"] == "run_all"]
    out = []
    for level, values in sens["risk_thresholds"].items():
        for v in values:
            e = copy.deepcopy(exp); e["risk"]["thresholds"][level] = v
            t = e["risk"]["thresholds"]
            if not t["moderate"] < t["high"] < t["very_high"]:
                continue
            name = "thresholds_" + "_".join(f"{k}{x:g}" for k, x in t.items())
            print(f"  {level} = {v}")
            sel = cached(name, lambda: cost(e, datasets, ALL_SCEN, modes=("selective",)))
            default = " (plan)" if v == exp["risk"]["thresholds"][level] else ""
            out += tag(exp4_metrics(sel, ra), "risk_thresholds", f"{level} = {v:g}{default}", v)
    return out


_MAIN: Dict[str, pd.DataFrame] = {}


def main_detection(exp, datasets) -> pd.DataFrame:
    if "det" not in _MAIN:
        _MAIN["det"] = cached("main_detection", lambda: detection(exp, datasets, ALL_SCEN))
    return _MAIN["det"]


def main_cost(exp, datasets) -> pd.DataFrame:
    if "cost" not in _MAIN:
        _MAIN["cost"] = cached("main_cost", lambda: cost(exp, datasets, ALL_SCEN))
    return _MAIN["cost"]


# ----------------------------------------------------------------------------- conclusions
def conclusions(long: pd.DataFrame, datasets) -> pd.DataFrame:
    """Re-check the main claims in every setting where they can be measured."""
    rows = []
    for (dial, setting), g in long.groupby(["dial", "setting"], sort=False):
        m = {(r.scope, r.metric): r.value for r in g.itertuples()}
        e1 = g[g.experiment == "exp1"]
        if len(e1) and any(s.endswith("|B_drift") for s in e1.scope):
            d_fpr = [m[(f"{d}|B_drift", "static_fpr")] - m[(f"{d}|B_drift", "adaptive_fpr")] for d in datasets]
            d_rec = [m[(f"{d}|B_drift", "static_recall")] - m[(f"{d}|B_drift", "adaptive_recall")] for d in datasets]
            rows.append({"dial": dial, "setting": setting, "claim": "C1 adaptive FPR < static FPR under drift (every dataset)",
                         "holds": all(x > 0 for x in d_fpr), "evidence": f"smallest FPR reduction {100 * min(d_fpr):.2f} pts"})
            rows.append({"dial": dial, "setting": setting, "claim": "C2 adaptive recall within 0.02 of static under drift",
                         "holds": all(x <= 0.02 for x in d_rec), "evidence": f"largest recall drop {max(d_rec):.3f}"})
        if ("ALL", "adaptive_accuracy_4class") in m:
            a, s = m[("ALL", "adaptive_accuracy_4class")], m[("ALL", "static_accuracy_4class")]
            rows.append({"dial": dial, "setting": setting, "claim": "C3 adaptive batch classification beats static",
                         "holds": a > s, "evidence": f"accuracy {a:.1%} vs {s:.1%}"})
        e4 = g[(g.experiment == "exp4") & (g.metric == "saving_data_pct") & (g.scope != "ALL|ALL")]
        if len(e4):
            rows.append({"dial": dial, "setting": setting, "claim": "C4 selective scans less data in every scenario",
                         "holds": bool((e4.value > 0).all()),
                         "evidence": f"saving {e4.value.min():.1f}% to {e4.value.max():.1f}%"})
            deg = [m[(f"ALL|{s}", "selective_recall")] for s in ("C_degradation", "D_drift_degradation")
                   if (f"ALL|{s}", "selective_recall") in m]
            if deg:
                rows.append({"dial": dial, "setting": setting, "claim": "C5 selective recall >= 0.90 under degradation",
                             "holds": min(deg) >= 0.90, "evidence": f"lowest recall {min(deg):.3f}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- tables and figures
def wide(long: pd.DataFrame, dial: str, keep) -> pd.DataFrame:
    g = long[(long.dial == dial) & long.apply(keep, axis=1)]
    t = g.pivot_table(index=["setting", "value_setting"], columns=["scope", "metric"], values="value", sort=False)
    t.columns = [f"{s}|{m}" for s, m in t.columns]
    return t.reset_index()


def seed_summary(long: pd.DataFrame) -> pd.DataFrame:
    g = long[long.dial == "seed"]
    s = g.groupby(["experiment", "scope", "metric"], sort=False)["value"].agg(["mean", "std", "min", "max", "count"])
    return s.reset_index().rename(columns={"std": "sd", "count": "seeds"})


def fig_exp1(long: pd.DataFrame, dial: str, xlab: str, path: Path, metrics=("fpr", "recall"),
             ylabels=("False-positive rate", "Recall")) -> None:
    g = long[(long.dial == dial) & (long.experiment == "exp1")]
    settings = list(dict.fromkeys(g.setting))
    ds_list = list(DATASET_LABELS)
    fig, axes = plt.subplots(len(metrics), len(ds_list), figsize=(12, 3.0 * len(metrics)), squeeze=False)
    for i, met in enumerate(metrics):
        for j, ds in enumerate(ds_list):
            ax = axes[i, j]
            for m in ("static", "adaptive"):
                s = g[(g.scope == f"{ds}|B_drift") & (g.metric == f"{m}_{met}")].set_index("setting").reindex(settings)
                ax.plot(range(len(settings)), s.value, color=COLORS[m], marker=MARKERS[m], markersize=5, linewidth=2,
                        label=LABELS[m])
            ax.set_xticks(range(len(settings)), [x.replace(" (", "\n(") for x in settings], fontsize=8)
            if i == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=10, color="#1f1f1c", loc="left")
            if i == len(metrics) - 1:
                ax.set_xlabel(xlab, fontsize=9, color="#55554f")
            if j == 0:
                ax.set_ylabel(ylabels[i], fontsize=9, color="#55554f")
            ax.set_ylim(bottom=0, top=1.05 if "recall" in met else None)
            style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.text(0.99, 0.01, "Scenario B (gradual drift 2% to 30%), evaluation batches T6-T20.", ha="right", fontsize=8,
             color="#55554f")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_z(long: pd.DataFrame, path: Path) -> None:
    g = long[(long.dial == "calibration_z") & (long.scope == "ALL")]
    settings = list(dict.fromkeys(g.setting))
    x = [float(s.split("=")[1]) for s in settings]
    series = [("adaptive_accuracy_4class", "4-class accuracy", "#2a78d6", "o"),
              ("adaptive_drift_detection_rate", "Drift detected", "#1baf7a", "^"),
              ("adaptive_drift_false_alarm_rate", "False drift alarms", "#eb6834", "s"),
              ("adaptive_degradation_detection_rate", "Degradation detected", "#7b61c9", "D")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for met, lab, col, mk in series:
        s = g[g.metric == met].set_index("setting").reindex(settings)
        ax.plot(x, 100 * s.value, color=col, marker=mk, markersize=6, linewidth=2, label=lab)
    ax.axvline(3.0, color="#9a9a93", linewidth=1, linestyle="--")
    ax.set_xlabel("Calibration z (threshold = warm-up mean + z sd)", fontsize=9, color="#55554f")
    ax.set_ylabel("% of evaluation batches", fontsize=9, color="#55554f")
    ax.set_ylim(-3, 103)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.text(0.99, 0.01, "Adaptive system, all datasets and scenarios, 180 batches. Dashed: default z = 3.",
             ha="right", fontsize=7.5, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_risk(long: pd.DataFrame, path: Path) -> None:
    """Every risk-weight and risk-threshold setting in the recall vs data-saved plane."""
    g = long[long.dial.isin(["risk_weights", "risk_thresholds"]) & (long.scope == "ALL|ALL")]
    p = g.pivot_table(index=["dial", "setting"], columns="metric", values="value", sort=False).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for dial, col, mk, lab in (("risk_weights", "#2a78d6", "o", "Risk weights varied"),
                               ("risk_thresholds", "#eb6834", "s", "Risk thresholds varied")):
        q = p[p.dial == dial]
        ax.scatter(q.saving_data_pct, q.selective_recall, s=40, color=col, marker=mk, label=lab, zorder=3)
    plan = p[p.setting.str.contains("plan")].iloc[0]
    ax.scatter([plan.saving_data_pct], [plan.selective_recall], s=170, facecolors="none", edgecolors="#1f1f1c",
               linewidths=1.3, zorder=4, label="Plan values (Sections 11, 12)")
    ax.axhline(p.run_all_recall.iloc[0], color="#9a9a93", linewidth=1, linestyle="--")
    ax.text(p.saving_data_pct.min(), p.run_all_recall.iloc[0] + 0.004, "run-all recall", fontsize=8, color="#55554f")
    for _, r in p.iterrows():   # label the two ends of the curve
        if r.saving_data_pct in (p.saving_data_pct.min(), p.saving_data_pct.max()):
            ax.annotate(r.setting, (r.saving_data_pct, r.selective_recall), textcoords="offset points",
                        xytext=(6, -3), fontsize=7.5, color="#55554f")
    ax.set_xlabel("Data scanned saved vs run-all (%)", fontsize=9, color="#55554f")
    ax.set_ylabel("Recall, risk-based selection", fontsize=9, color="#55554f")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.text(0.99, 0.01, "All datasets and scenarios, T6-T20. One setting changed at a time.", ha="right",
             fontsize=7.5, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    global FORCE, KEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--dial", nargs="*", choices=DIALS, default=DIALS)
    ap.add_argument("--force", action="store_true", help="recompute settings already cached in results/sensitivity/raw")
    ap.add_argument("--keep-data", action="store_true", help="keep the rebuilt batches in data/sensitivity_work")
    args = ap.parse_args()
    FORCE, KEEP = args.force, args.keep_data
    t0 = time.perf_counter()
    datasets, exp = datasets_config(), experiment_config()
    sens = exp["sensitivity"]
    for d in (RAW, TABLES, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    fns = {"seed": dial_seed, "defect_rate": dial_defect_rate, "outlier_severity": dial_outlier_severity,
           "calibration_z": dial_calibration_z, "risk_weights": dial_risk_weights, "risk_thresholds": dial_risk_thresholds}
    long_path = TABLES / "sens_all_results_long.csv"
    previous = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame(columns=["dial"])
    parts = [previous[~previous.dial.isin(args.dial)]]
    for dial in args.dial:
        print(f"[{dial}]")
        parts.append(pd.DataFrame(fns[dial](exp, datasets, sens)))
    long = pd.concat([p for p in parts if len(p)], ignore_index=True)
    long["dial"] = pd.Categorical(long.dial, DIALS, ordered=True)
    long = long.sort_values("dial", kind="stable")
    long["dial"] = long.dial.astype(str)
    long.to_csv(long_path, index=False)

    # readable tables per dial
    dials = set(long.dial)
    key_exp1 = lambda r: r.experiment == "exp1" and r.scope.endswith("|B_drift") and \
        r.metric.split("_", 1)[1] in ("fpr", "recall", "f1", "recall_outlier")  # noqa: E731
    key_exp4 = lambda r: r.experiment == "exp4" and r.metric in ("selective_recall", "saving_data_pct")  # noqa: E731
    if "seed" in dials:
        seed_summary(long).to_csv(TABLES / "sens_seeds_summary.csv", index=False)
    if "defect_rate" in dials:
        wide(long, "defect_rate", key_exp1).to_csv(TABLES / "sens_defect_rate_exp1.csv", index=False)
        wide(long, "defect_rate", key_exp4).to_csv(TABLES / "sens_defect_rate_exp4.csv", index=False)
        fig_exp1(long, "defect_rate", "Defect rate (share of records)", FIGS / "sens_defect_rate.png")
    if "outlier_severity" in dials:
        wide(long, "outlier_severity", key_exp1).to_csv(TABLES / "sens_outlier_severity.csv", index=False)
        fig_exp1(long, "outlier_severity", "Outlier distance from the mean", FIGS / "sens_outlier_severity.png",
                 metrics=("recall_outlier", "fpr"), ylabels=("Outlier recall", "False-positive rate"))
    if "calibration_z" in dials:
        wide(long, "calibration_z", lambda r: r.experiment == "exp2").to_csv(TABLES / "sens_calibration_z.csv", index=False)
        fig_z(long, FIGS / "sens_calibration_z.png")
    for dial in ("risk_weights", "risk_thresholds"):
        if dial in dials:
            wide(long, dial, key_exp4).to_csv(TABLES / f"sens_{dial}.csv", index=False)
    if {"risk_weights", "risk_thresholds"} <= dials:
        fig_risk(long, FIGS / "sens_risk_tradeoff.png")
    concl = conclusions(long, datasets)
    concl.to_csv(TABLES / "sens_conclusions.csv", index=False)

    verdict = concl.groupby("claim").holds.agg(["sum", "count"]).reset_index()
    stamp = now_iso().replace(":", "").replace("-", "")
    log = {"experiment_id": f"sensitivity_{stamp}", "date": now_iso(), "random_seed": exp["seed"],
           "dials_run": args.dial, "settings": sens,
           "defaults": {"defect_rate": exp["schedules"]["defects_baseline"][0], "outlier_sigma": exp["defects"]["outlier_sigma"],
                        "calibration_z": exp["classification"]["calibration_z"], "risk": exp["risk"]},
           "conclusions": {r.claim: f"holds in {r['sum']} of {r['count']} settings" for _, r in verdict.iterrows()},
           "runtime_s": time.perf_counter() - t0, "cloud_cost": "none (local run)",
           "output_paths": {"tables": str(TABLES), "figures": str(FIGS), "cache": str(RAW)}}
    log_path = results_dir("raw", "run_logs", f"{log['experiment_id']}.json")
    write_json(log, log_path)

    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
    print()
    print(concl.to_string(index=False))
    print()
    for claim, v in log["conclusions"].items():
        print(f"{claim}: {v}")
    print(f"\nRun log: {log_path}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
