"""Evaluation metrics (project description, Section 20)."""
from __future__ import annotations

from typing import Dict, Iterable, Set


def confusion(flagged: Set[str], defective: Set[str], all_ids: Iterable[str]) -> Dict[str, int]:
    all_ids = set(all_ids)
    tp = len(flagged & defective)
    fp = len(flagged - defective)
    fn = len(defective - flagged)
    tn = len(all_ids) - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def scores(c: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }
