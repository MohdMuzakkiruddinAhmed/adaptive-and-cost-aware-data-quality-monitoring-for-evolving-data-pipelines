"""Run a monitoring configuration over the batches of one scenario, with cost measurement.

mode "run_all"    adaptive thresholds, every check on every batch (Experiment 4 baseline)
mode "selective"  adaptive thresholds, checks chosen per batch by risk (Sections 11-12)
mode "static"     static thresholds (T1), every check on every batch
mode "fixed"      only the rules passed in ``extra_rules`` (e.g. validated LLM rules), every batch
``extra_rules`` are appended to the rule set in the other modes (e.g. adaptive + LLM rules).

T1..T5 are warm-up and always run every check. The failure-rate baselines of the risk
score (F) and the no-drift thresholds of its drift term (D) are calibrated on them.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adaptation.adaptive_thresholds import adaptive_category_rules, adaptive_range_rules
from common.metrics import confusion, scores
from common.utils import batch_label, data_dir
from drift.drift_detector import profile_drift_scores, reference_summary
from execution.local_executor import column_bytes, execute
from experiments.detection import _type_recall, load_batch, static_rule_set
from optimizer.rule_selector import select
from profiling.store import ProfileStore
from risk.risk_score import RECORDS, RiskTracker, calibrate_f
from rules.static_rules import base_rules


def adaptive_rule_set(columns: List[str], history, current, cfg, exp) -> List[Dict[str, Any]]:
    num, cat = cfg["numeric"], cfg["categorical"]
    return (base_rules(columns, num, cat)
            + adaptive_category_rules(history, cat, exp["adaptive"]["category_min_batches"])
            + adaptive_range_rules(history, current, {}, num, exp["range_k"]))


def _drift_thresholds(warm_profiles, cfg, z=3.0) -> Dict[str, float]:
    """No-drift threshold per column: leave-one-out profile drift over the warm-up batches."""
    rows = []
    for j, p in enumerate(warm_profiles):
        others = [q for i, q in enumerate(warm_profiles) if i != j]
        rows.append(profile_drift_scores(p, others, cfg["numeric"], cfg["categorical"]))
    df = pd.DataFrame(rows)
    return {c: max(float(df[c].mean() + z * df[c].std(ddof=1)), 1e-6) for c in df.columns}


def run(dataset: str, scenario: str, cfg: Dict[str, Any], exp: Dict[str, Any], mode: str,
        moderate: Optional[float] = None, extra_rules: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rcfg = exp["risk"]
    thr = dict(rcfg["thresholds"])
    if moderate is not None:
        thr["moderate"] = moderate
    warm, window = exp["warmup_batches"], exp["adaptive"]["history_window"]
    store = ProfileStore(dataset, scenario)
    labels = pd.read_csv(data_dir("ground_truth", dataset, scenario, "labels.csv")).set_index("batch")
    t1 = store.load(batch_label(1))
    units = cfg["numeric"] + cfg["categorical"] + [RECORDS]
    baseline = store.history([batch_label(i) for i in range(1, warm + 1)])   # warm-up reference, loaded once
    baseline_ref = reference_summary(baseline, cfg["numeric"], cfg["categorical"])

    warm_rates: List[Dict[str, float]] = []
    tracker: Optional[RiskTracker] = None
    last_run: Dict[str, int] = {}
    prev_schema = None
    rows = []

    for t in range(1, exp["n_batches"] + 1):
        b, df, gt = load_batch(dataset, scenario, t)
        current = store.load(b)
        cols = list(df.columns)
        if mode == "fixed":
            rules = list(extra_rules or [])
        elif mode == "static" or t == 1:
            rules = static_rule_set(t1, cols, cfg, exp["range_k"]) + list(extra_rules or [])
        else:
            history = store.history([batch_label(i) for i in range(max(1, t - window), t)])
            rules = adaptive_rule_set(cols, history, current, cfg, exp) + list(extra_rules or [])
        cb = column_bytes(df)

        info: Dict[str, Any] = {}
        overhead_ms = 0.0
        if mode == "selective" and t > warm:
            t0 = time.perf_counter()
            d_raw = profile_drift_scores(current, baseline_ref, cfg["numeric"], cfg["categorical"])
            schema = current["dataset"]["schema"]
            changed = {u: (u not in schema) or (prev_schema is not None and prev_schema.get(u) != schema.get(u))
                       for u in units if u != RECORDS}
            risk = tracker.score(d_raw, changed)
            chosen, info = select(rules, risk, rcfg["tiers"], thr, last_run, t, rcfg["refresh_every"])
            overhead_ms = (time.perf_counter() - t0) * 1000
        else:
            chosen, risk = rules, None

        res = execute(chosen, df, cb)
        n = len(df)
        observed = {p["key"]: p["failures"] / n for p in res["per_rule"]}
        for p in res["per_rule"]:
            last_run[p["key"]] = t

        if mode == "selective":
            if t <= warm:
                if t >= 2:
                    warm_rates.append(observed)
                if t == warm:
                    tracker = RiskTracker(units, cfg["importance"], rcfg["weights"], calibrate_f(warm_rates, n),
                                          _drift_thresholds(baseline, cfg), rcfg["history_window"])
            else:
                tracker.update(observed, risk)
        prev_schema = current["dataset"]["schema"]

        c = confusion(res["failed_ids"], set(gt["record_id"]), df["record_id"])
        row = {"dataset": dataset, "scenario": scenario, "batch": b, "t": t, "mode": mode,
               "moderate_threshold": thr["moderate"], "evaluation": int(labels.loc[b, "evaluation"]),
               "drift_level": labels.loc[b, "drift_level"], "defect_rate": labels.loc[b, "defect_rate"],
               "records": n, "defects": len(gt), **c, **scores(c), **_type_recall(res["failed_ids"], gt),
               "rules_available": len(rules), "rules_executed": res["rules_executed"],
               "runtime_ms": res["runtime_ms"] + overhead_ms, "selection_overhead_ms": overhead_ms,
               "bytes_scanned": res["bytes_scanned"]}
        if risk is not None:
            row["max_risk"] = max(v["R"] for v in risk.values())
            row["complete_suite"] = int(info["complete_suite"])
            for lv in ("low", "moderate", "high", "very_high"):
                row[f"units_{lv}"] = sum(1 for x in info["levels"].values() if x == lv)
            row.update({f"checks_{k}": info[k] for k in ("basic", "risk", "refresh", "very_high", "skipped")})
            row.update({f"R_{u}": v["R"] for u, v in risk.items()})
        rows.append(row)
    return rows
