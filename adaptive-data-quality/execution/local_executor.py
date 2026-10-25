"""Execute a rule set on a batch and record which records fail and what it cost
(Section 11: execution time, data scanned, number of records, rule type).

Profile reuse (manuscript Sections 2.5 and 5.7; supervisor: "let baselines reuse already computed
statistics"). The batch profile already contains each column's missing share, min/max, category
frequencies and, in memory, the whole-record duplicate mask. With ``reuse`` given, a check whose outcome
is decided by those statistics does not rescan the column:

  completeness / type  no missing (or non-numeric) value in the profile -> no failures, no scan
  range                profile min and max inside the bounds -> no failures, no scan
  category / pattern   evaluated on the DISTINCT values in the profile; the column is scanned only to
                       mark the records holding a disallowed value (if any)
  uniqueness           the duplicate mask computed while profiling is reused

Results are identical with and without reuse (tests/test_monitor.py); only cost differs. The logical
byte-touch estimate ("bytes_scanned") counts a column only when it is actually scanned.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rules.static_rules import check, describe


def column_bytes(df: pd.DataFrame) -> Dict[str, int]:
    return {c: int(v) for c, v in df.memory_usage(deep=True, index=False).items()}


def _scanned(rule: Dict[str, Any], col_bytes: Dict[str, int]) -> int:
    from rules.static_rules import involved_columns
    return sum(col_bytes.get(c, 0) for c in involved_columns(rule))


def _key(rule: Dict[str, Any]) -> str:
    return rule.get("key") or f"{rule['type']}:{rule.get('column') or '_records'}"


def check_with_reuse(rule: Dict[str, Any], df: pd.DataFrame, reuse: Dict[str, Any]):
    """(failure mask, scanned?) using profile statistics where they decide the outcome."""
    prof = reuse.get("profile") or {}
    cols = prof.get("columns", {})
    t = rule["type"]
    n = len(df)
    if t in ("dependency", "conditional"):                 # cross-column: no profile statistic decides them
        return check(rule, df).fillna(False).to_numpy(dtype=bool), True
    if t == "uniqueness" and reuse.get("dup_mask") is not None and \
            list(rule["columns"]) == list(reuse.get("dup_columns") or []):
        return np.asarray(reuse["dup_mask"], dtype=bool), False
    c = rule.get("column")
    p = cols.get(c)
    if p is None:
        return check(rule, df).fillna(False).to_numpy(dtype=bool), True
    if t in ("completeness", "type") and (p.get("missing_pct") in (0, 0.0)):
        return np.zeros(n, dtype=bool), False
    if t == "range" and p.get("datatype") == "numerical" and p.get("count"):
        lo = -np.inf if rule.get("low") is None else rule["low"]
        hi = np.inf if rule.get("high") is None else rule["high"]
        if lo <= p["min"] and p["max"] <= hi:          # NaN never fails a range rule
            return np.zeros(n, dtype=bool), False
    if t in ("category", "pattern") and "frequencies" in p:
        values = list(p["frequencies"])
        if t == "category":
            allowed = set(rule["allowed"])
            bad = [v for v in values if v not in allowed]
        else:
            rx = re.compile(rule["pattern"])
            bad = [v for v in values if not rx.fullmatch(v)]
        if not bad:
            return np.zeros(n, dtype=bool), False
        s = df[c]
        return (s.notna() & s.astype(str).isin(set(bad))).to_numpy(dtype=bool), True
    return check(rule, df).fillna(False).to_numpy(dtype=bool), True


def execute(rules: List[Dict[str, Any]], df: pd.DataFrame, col_bytes: Optional[Dict[str, int]] = None,
            reuse: Optional[Dict[str, Any]] = None, return_masks: bool = False) -> Dict[str, Any]:
    col_bytes = col_bytes if col_bytes is not None else column_bytes(df)
    failed = np.zeros(len(df), dtype=bool)
    per_rule = []
    masks: Dict[str, np.ndarray] = {}
    t_all = time.perf_counter()
    for r in rules:
        t0 = time.perf_counter()
        if reuse is not None:
            mask, scanned = check_with_reuse(r, df, reuse)
        else:
            mask, scanned = check(r, df).fillna(False).to_numpy(dtype=bool), True
        key = _key(r)
        per_rule.append({"rule": describe(r), "type": r["type"], "column": r.get("column"), "key": key,
                         "source": r.get("source"), "failures": int(mask.sum()),
                         "runtime_ms": (time.perf_counter() - t0) * 1000,
                         "bytes_scanned": _scanned(r, col_bytes) if scanned else 0, "scanned": bool(scanned)})
        failed |= mask
        if return_masks:
            masks[key] = mask
    out = {
        "failed_ids": set(df["record_id"].to_numpy()[failed]),
        "failure_rate": float(failed.mean()) if len(df) else 0.0,
        "rules_executed": len(rules),
        "records_scanned": len(df),
        "bytes_scanned": sum(p["bytes_scanned"] for p in per_rule),
        "runtime_ms": (time.perf_counter() - t_all) * 1000,
        "per_rule": per_rule,
    }
    if return_masks:
        out["masks"] = masks
        out["failed_mask"] = failed
    return out
