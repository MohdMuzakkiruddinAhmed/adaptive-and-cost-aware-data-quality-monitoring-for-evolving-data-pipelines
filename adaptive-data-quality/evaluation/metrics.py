"""Record-, batch- and incident-level measures for the v2 experiments (manuscript Section 6.5).

Record level (per batch, summed over batches and seeds before ratios are formed):
  TP, FP, FN, TN; P = TP + FN (defective records); N = FP + TN (valid records)
  breakdown by defect type, defect type x severity, and affected column (tp_* / n_* count columns)
  operational: false positives per 10,000 valid records, missed errors per 10,000 records
Batch labels: four-class macro-F1, balanced accuracy, per-class precision/recall, confusion matrix,
  false degradation alarms during pure drift, detection delay after onset
Incidents: time to first detection, post-onset recall, never-detected incidents, recovery time
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

CLASSES = ["stable", "drift", "degradation", "drift+degradation"]


def record_counts(flagged: set, gt: pd.DataFrame, all_ids: Iterable[str],
                  valid_novel_ids: Optional[set] = None) -> Dict[str, Any]:
    all_ids = set(all_ids)
    defective = set(gt["record_id"])
    tp = len(flagged & defective)
    fp = len(flagged - defective)
    fn = len(defective - flagged)
    tn = len(all_ids) - tp - fp - fn
    out: Dict[str, Any] = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "P": tp + fn, "N": fp + tn, "records": len(all_ids)}
    if len(gt):
        for t, g in gt.groupby("defect_type"):
            ids = set(g["record_id"])
            out[f"n_type:{t}"] = len(ids)
            out[f"tp_type:{t}"] = len(ids & flagged)
        for (t, s), g in gt.groupby(["defect_type", "severity"]):
            ids = set(g["record_id"])
            out[f"n_sev:{t}|{s}"] = len(ids)
            out[f"tp_sev:{t}|{s}"] = len(ids & flagged)
        for c, g in gt.groupby("column"):
            ids = set(g["record_id"])
            out[f"n_col:{c}"] = len(ids)
            out[f"tp_col:{c}"] = len(ids & flagged)
        if "incident" in gt.columns:
            for inc, g in gt[gt["incident"].astype(str) != ""].groupby("incident"):
                ids = set(g["record_id"])
                out[f"n_inc:{inc}"] = len(ids)
                out[f"tp_inc:{inc}"] = len(ids & flagged)
    if valid_novel_ids is not None:
        out["n_valid_novel"] = len(valid_novel_ids)
        out["fp_valid_novel"] = len(valid_novel_ids & flagged)
    return out


def ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add precision, recall, F1, FPR and operational rates to a frame of summed counts."""
    d = df.copy()
    d["precision"] = d.TP / (d.TP + d.FP).where(lambda s: s > 0)
    d["recall"] = d.TP / d.P.where(lambda s: s > 0)
    d["f1"] = 2 * d.precision * d.recall / (d.precision + d.recall)
    d["fpr"] = d.FP / d.N.where(lambda s: s > 0)
    d["fp_per_10k_valid"] = 1e4 * d["fpr"]
    d["missed_per_10k_records"] = 1e4 * d.FN / d.records.where(lambda s: s > 0)
    return d


def breakdown_long(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Recall by defect type / severity / column / incident in long form, from tp_* and n_* columns."""
    rows = []
    n_cols = [c for c in df.columns if c.startswith("n_") and ":" in c]
    sums = df.groupby(by)[n_cols + [c.replace("n_", "tp_", 1) for c in n_cols]].sum().reset_index()
    for _, r in sums.iterrows():
        for c in n_cols:
            kind, name = c[2:].split(":", 1)
            n, tp = r[c], r[c.replace("n_", "tp_", 1)]
            if n:
                rows.append({**{k: r[k] for k in by}, "breakdown": kind, "group": name, "n": int(n),
                             "detected": int(tp), "recall": tp / n})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- batch labels
def label_metrics(true: Sequence[str], pred: Sequence[str]) -> Dict[str, Any]:
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    present = [c for c in CLASSES if (true == c).any()]
    f1s, recs = [], []
    out: Dict[str, Any] = {"batches": len(true), "accuracy": float((true == pred).mean()) if len(true) else np.nan}
    for c in CLASSES:
        tp = int(((true == c) & (pred == c)).sum())
        fp = int(((true != c) & (pred == c)).sum())
        fn = int(((true == c) & (pred != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else np.nan
        f = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and (p + r) else 0.0
        out[f"precision:{c}"], out[f"recall:{c}"], out[f"f1:{c}"] = p, r, f
        if c in present:
            f1s.append(f)
            recs.append(r)
    out["macro_f1"] = float(np.mean(f1s)) if f1s else np.nan
    out["balanced_accuracy"] = float(np.mean(recs)) if recs else np.nan
    pure_drift = true == "drift"
    out["false_degradation_during_drift"] = float(np.isin(pred[pure_drift], ["degradation", "drift+degradation"]).mean()) \
        if pure_drift.any() else np.nan
    out["insufficient_share"] = float((pred == "insufficient").mean()) if len(pred) else np.nan
    return out


def confusion(true: Sequence[str], pred: Sequence[str]) -> pd.DataFrame:
    cols = CLASSES + ["insufficient"]
    cm = pd.crosstab(pd.Categorical(true, CLASSES), pd.Categorical(pred, cols), dropna=False)
    return cm.reindex(index=CLASSES, columns=cols, fill_value=0)


def onset_delay(ts: Sequence[int], flags: Sequence[bool], onset: Optional[int]) -> Optional[float]:
    """Batches from onset to the first flagged batch at or after onset (0 = flagged at onset); NaN if never."""
    if onset is None:
        return None
    for t, f in sorted(zip(ts, flags)):
        if t >= onset and f:
            return float(t - onset)
    return float("nan")


# ----------------------------------------------------------------------------- incidents
def incident_summary(per_batch: pd.DataFrame, name: str, start: int, end: int, detect_recall: float,
                     n_batches: int) -> Dict[str, Any]:
    """``per_batch``: rows of one run (t, n_inc:<name>, tp_inc:<name>, label) for the evaluation batches."""
    n_col, tp_col = f"n_inc:{name}", f"tp_inc:{name}"
    pb = per_batch.sort_values("t")
    inside = pb[(pb.t >= start) & (pb.t <= end)]
    n = inside.get(n_col, pd.Series(0, index=inside.index)).fillna(0)
    tp = inside.get(tp_col, pd.Series(0, index=inside.index)).fillna(0)
    rec = (tp / n.where(n > 0)).fillna(0)
    detected_t = inside.t[rec >= detect_recall]
    label_deg = pb["label"].astype(str).str.contains("degradation") if "label" in pb else pd.Series(False, index=pb.index)
    lab_in = pb[(pb.t >= start) & label_deg]
    after = pb[pb.t > end]
    rec_t = after.t[~label_deg.loc[after.index]] if len(after) else pd.Series(dtype=int)
    return {"incident": name, "start": start, "end": end,
            "post_onset_recall": float(tp.sum() / n.sum()) if n.sum() else np.nan,
            "detection_delay": float(detected_t.iloc[0] - start) if len(detected_t) else np.nan,
            "detected": bool(len(detected_t)),
            "label_delay": float(lab_in.t.iloc[0] - start) if len(lab_in) else np.nan,
            "recovery_time": (float(rec_t.iloc[0] - end) if len(rec_t) else np.nan) if end < n_batches else np.nan,
            "false_alarms_after_recovery": int(label_deg.loc[after.index].sum()) if len(after) else 0}
