"""Run the static and the adaptive monitoring method over every batch of a scenario.

static    rules fixed from the T1 profile: completeness, uniqueness, type,
          category (categories seen in T1), range mu +/- 3 sigma (T1).
adaptive  the same rule types, recomputed each batch from the last k profiles and the
          current drift score: T_t = f(H_{t-k:t}, D_t)  (adaptation/adaptive_thresholds.py).

Drift scores are computed on records that pass the completeness, uniqueness and category
rules and a coarse 6-sigma gate, so injected defects do not masquerade as drift.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from adaptation.adaptive_thresholds import adaptive_category_rules, adaptive_range_rules, wide_gate_rules
from common.metrics import confusion, scores
from common.utils import batch_label, data_dir
from drift.drift_detector import drift_scores
from execution.local_executor import execute
from profiling.store import ProfileStore
from rules.static_rules import base_rules
from rules.statistical_rules import category_rules, range_rules

DEFECT_TYPES = ["missing", "duplicate", "invalid_category", "outlier", "semantic_numeric", "semantic_category"]


def load_batch(dataset: str, scenario: str, t: int) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    b = batch_label(t)
    df = pd.read_parquet(data_dir("batches", dataset, scenario, f"{b}.parquet"))
    gt = pd.read_csv(data_dir("ground_truth", dataset, scenario, f"{b}.csv"))
    return b, df, gt


def static_rule_set(reference: Dict[str, Any], columns: List[str], cfg: Dict[str, Any], k: float) -> List[Dict]:
    num, cat = cfg["numeric"], cfg["categorical"]
    return base_rules(columns, num, cat) + range_rules(reference, num, k) + category_rules(reference, cat)


def adaptive_batch(df: pd.DataFrame, current: Dict[str, Any], history: List[Dict[str, Any]],
                   baseline: List[Dict[str, Any]], cfg: Dict[str, Any], exp: Dict[str, Any]):
    """Adaptive monitoring for one batch. Returns (execution result, rolling drift, baseline drift)."""
    num, cat = cfg["numeric"], cfg["categorical"]
    base = base_rules(list(df.columns), num, cat)
    cats = adaptive_category_rules(history, cat, exp["adaptive"]["category_min_batches"])
    gate = execute(base + cats + wide_gate_rules(history, num), df)
    passing = df[~df["record_id"].isin(gate["failed_ids"])]
    d_roll = drift_scores(passing, history, num, cat)        # how far to move the thresholds
    d_base = drift_scores(passing, baseline, num, cat)       # has the data drifted? (Experiment 2)
    res = execute(base + cats + adaptive_range_rules(history, current, d_roll, num, exp["range_k"]), df)
    res["runtime_ms"] += gate["runtime_ms"]
    return res, d_roll, d_base


def _type_recall(flagged: set, gt: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for t in DEFECT_TYPES:
        ids = set(gt.loc[gt["defect_type"] == t, "record_id"])
        out[f"n_{t}"] = len(ids)
        out[f"tp_{t}"] = len(ids & flagged)
    for sev in ("Medium", "High"):
        ids = set(gt.loc[(gt["defect_type"] == "outlier") & (gt["severity"] == sev), "record_id"])
        out[f"n_outlier_{sev}"] = len(ids)
        out[f"tp_outlier_{sev}"] = len(ids & flagged)
    sem = gt[gt["defect_type"].astype(str).str.startswith("semantic")]
    for col, g in sem.groupby("column"):          # semantic defects: recall per violated column
        ids = set(g["record_id"])
        out[f"n_sem_{col}"] = len(ids)
        out[f"tp_sem_{col}"] = len(ids & flagged)
    return out


def run_scenario(dataset: str, scenario: str, cfg: Dict[str, Any], exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    window, warm = exp["adaptive"]["history_window"], exp["warmup_batches"]
    store = ProfileStore(dataset, scenario)
    labels = pd.read_csv(data_dir("ground_truth", dataset, scenario, "labels.csv")).set_index("batch")
    t1 = store.load(batch_label(1))
    rows: List[Dict[str, Any]] = []

    for t in range(1, exp["n_batches"] + 1):
        b, df, gt = load_batch(dataset, scenario, t)
        defective = set(gt["record_id"])
        common = {"dataset": dataset, "scenario": scenario, "batch": b, "t": t,
                  **{k: labels.loc[b, k] for k in ("drift_level", "defect_rate", "drift_label",
                                                    "degradation_label", "evaluation")},
                  "records": len(df), "defects": len(defective)}

        res = execute(static_rule_set(t1, list(df.columns), cfg, exp["range_k"]), df)
        c = confusion(res["failed_ids"], defective, df["record_id"])
        rows.append({**common, "method": "static", **c, **scores(c), **_type_recall(res["failed_ids"], gt),
                     "failure_rate": res["failure_rate"], "rules_executed": res["rules_executed"],
                     "runtime_ms": res["runtime_ms"]})

        if t == 1:
            res_a, d_base = res, {}
        else:
            history = store.history([batch_label(i) for i in range(max(1, t - window), t)])
            baseline = store.history([batch_label(i) for i in range(1, min(t, warm + 1))])
            res_a, _, d_base = adaptive_batch(df, store.load(b), history, baseline, cfg, exp)
        c = confusion(res_a["failed_ids"], defective, df["record_id"])
        rows.append({**common, "method": "adaptive", **c, **scores(c), **_type_recall(res_a["failed_ids"], gt),
                     "failure_rate": res_a["failure_rate"], "rules_executed": res_a["rules_executed"],
                     "runtime_ms": res_a["runtime_ms"], **{f"D_{col}": v for col, v in d_base.items()}})
    return rows


def run_all(datasets: Dict[str, Any], exp: Dict[str, Any]) -> pd.DataFrame:
    out = []
    for name, cfg in datasets.items():
        for scen in exp["scenarios"]:
            out += run_scenario(name, scen, cfg, exp)
    return pd.DataFrame(out)


def warmup_calibration(dataset: str, scenario: str, cfg: Dict[str, Any], exp: Dict[str, Any]) -> pd.DataFrame:
    """Leave-one-out over the warm-up batches: each of T1..T5 is scored with the other four
    as history and baseline. Gives the normal level and noise of the drift scores and
    failure rate (static and adaptive) at full history size, used to set Experiment 2 thresholds."""
    warm = exp["warmup_batches"]
    store = ProfileStore(dataset, scenario)
    rows = []
    for t in range(1, warm + 1):
        b, df, _ = load_batch(dataset, scenario, t)
        others = [store.load(batch_label(i)) for i in range(1, warm + 1) if i != t]
        res, _, d_base = adaptive_batch(df, store.load(b), others, others, cfg, exp)
        ref = others[0]  # static rules come from a single reference batch
        res_s = execute(static_rule_set(ref, list(df.columns), cfg, exp["range_k"]), df)
        rows.append({"batch": b, "failure_rate": res["failure_rate"], "static_failure_rate": res_s["failure_rate"],
                     **{f"D_{c}": v for c, v in d_base.items()}})
    return pd.DataFrame(rows)
