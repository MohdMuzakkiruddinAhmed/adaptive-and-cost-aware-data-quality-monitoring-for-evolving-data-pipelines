"""Rule-level analysis for RQ2 / H2a (supervisor experiment 3; manuscript Sections 5.2 and 6.5).

admission_outcomes   every LLM candidate rule evaluated on the EVALUATION batches (ground truth used only
                     here, after admission): false-positive rate on valid records, semantic errors caught,
                     semantic errors caught that the statistical suite missed ("unique"). Then:
                       incorrect rule        flags more than ``fp_tolerance`` of valid records
                       incorrectly admitted  admitted (accepted/modified) but incorrect
                       incorrectly rejected  rejected or removed as redundant, although correct and it
                                             uniquely catches semantic errors
predicate_stability  agreement of the generated PREDICATES between generations (not only rule types):
                     each generation's rules on a column are applied to a common probe set of values (real
                     training values plus synthetic edge values), and two generations agree on a value when
                     both accept or both reject it. Also interval Jaccard for ranges and set Jaccard for
                     category lists.
"""
from __future__ import annotations

import re
import string
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from rules.static_rules import check
from rules.validator import normalise

ADMITTED = ("accepted", "modified")


def admission_outcomes(validation: List[Dict[str, Any]], candidates: List[Dict[str, Any]], cfg: Dict[str, Any],
                       eval_batches: Sequence[tuple], fp_tolerance: float) -> pd.DataFrame:
    """``eval_batches``: (df, gt, statistical_flagged_ids) for every evaluation batch."""
    rows = []
    for v in validation:
        if v["status"] == "invalid":
            rows.append({"index": v["index"], "status": "invalid", "column": v.get("column"),
                         "rule_type": v.get("rule_type"), "admitted": False})
            continue
        # a "modified" rule is admitted (and executed) in its relaxed form, so it is judged in that form; rejected and
        # redundant candidates are judged as written (v2.2: before, a relaxed category list was judged as written and
        # counted as "incorrectly admitted" although the executed rule raised no false alarms)
        if v["status"] == "modified" and v.get("rule") is not None:
            rule = dict(v["rule"])
        else:
            rule, _ = normalise(candidates[v["index"]], cfg, list(eval_batches[0][0].columns))
        fp = valid = sem_tp = sem_n = uniq = 0
        for df, gt, stat_flagged in eval_batches:
            mask = check(rule, df).fillna(False).to_numpy(dtype=bool)
            ids = df["record_id"].to_numpy()
            flagged = set(ids[mask])
            defective = set(gt["record_id"])
            sem = set(gt.loc[gt["defect_type"].astype(str).str.startswith("semantic"), "record_id"])
            valid += len(df) - len(defective & set(ids))
            fp += len(flagged - defective)
            sem_n += len(sem)
            sem_tp += len(flagged & sem)
            uniq += len((flagged & sem) - set(stat_flagged))
        fpr = fp / valid if valid else np.nan
        correct = bool(fpr <= fp_tolerance)
        admitted = v["status"] in ADMITTED
        rows.append({"index": v["index"], "status": v["status"], "column": rule.get("column"), "rule_type": rule["type"],
                     "admitted": admitted, "fp_rate_valid": fpr, "semantic_caught": sem_tp,
                     "semantic_total": sem_n, "unique_semantic_caught": uniq, "correct": correct,
                     "incorrectly_admitted": admitted and not correct,
                     "incorrectly_rejected": (not admitted) and correct and uniq > 0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- stability
def probe_values(df: pd.DataFrame, column: str, numeric: bool, n: int, seed: int) -> np.ndarray:
    """Common probe set: sampled real values plus deterministic synthetic edge values."""
    rng = np.random.default_rng(seed)
    vals = df[column].dropna().to_numpy()
    real = vals[rng.choice(len(vals), size=min(n, len(vals)), replace=False)] if len(vals) else vals
    if numeric:
        x = real.astype(float)
        span = (np.nanmax(x) - np.nanmin(x)) if len(x) else 1.0
        edges = np.array([0.0, -1.0, -span, np.nanmin(x) - 0.5 * span, np.nanmax(x) + 0.5 * span,
                          10 * np.nanmax(np.abs(x))]) if len(x) else np.array([0.0, -1.0])
        return np.concatenate([x, edges])
    real = real.astype(str)
    letters = np.array(list(string.ascii_uppercase))
    synth = []
    for v in real[: max(1, n // 4)]:
        synth.append("".join(rng.choice(letters, len(v))) if v.isupper() else v[::-1])   # well-formed / scrambled
        if len(v) > 2:
            synth.append(v[0] + v[2] + v[1] + v[3:])                                      # transposition typo
    return np.concatenate([real, np.array(synth, dtype=object)])


def _decisions(rules: List[Dict[str, Any]], column: str, probes: np.ndarray) -> np.ndarray:
    """True where a probe value fails ANY of the given rules on ``column``."""
    df = pd.DataFrame({column: probes})
    fail = np.zeros(len(probes), dtype=bool)
    for r in rules:
        fail |= check(r, df).fillna(False).to_numpy(dtype=bool)
    return fail


def _interval_jaccard(a, b, lo_ref: float, hi_ref: float) -> float:
    def clip(lo, hi):
        lo = lo_ref if lo is None or not np.isfinite(lo) else max(lo, lo_ref)
        hi = hi_ref if hi is None or not np.isfinite(hi) else min(hi, hi_ref)
        return lo, hi
    (la, ha), (lb, hb) = clip(*a), clip(*b)
    inter = max(0.0, min(ha, hb) - max(la, lb))
    union = max(ha, hb) - min(la, lb)
    return inter / union if union > 0 else 1.0


def predicate_stability(rule_sets: Dict[Any, List[Dict[str, Any]]], probe_df: pd.DataFrame, cfg: Dict[str, Any],
                        n_probe: int = 200, seed: int = 0) -> pd.DataFrame:
    """Pairwise agreement between generations. ``rule_sets``: generation id -> executable rules."""
    cols = cfg["numeric"] + cfg["categorical"]
    rows = []
    for c in cols:
        numeric = c in cfg["numeric"]
        probes = probe_values(probe_df, c, numeric, n_probe, seed)
        x = probe_df[c].dropna()
        lo_ref, hi_ref = (float(x.min()) - abs(float(x.max()) - float(x.min())), float(x.max()) * 2 + 1) if numeric else (0, 1)
        for (ga, ra), (gb, rb) in combinations(sorted(rule_sets.items(), key=lambda kv: str(kv[0])), 2):
            ca = [r for r in ra if r.get("column") == c and r["type"] in ("range", "category", "pattern")]
            cb = [r for r in rb if r.get("column") == c and r["type"] in ("range", "category", "pattern")]
            da, db = _decisions(ca, c, probes), _decisions(cb, c, probes)
            row = {"column": c, "gen_a": ga, "gen_b": gb, "probe_agreement": float((da == db).mean()),
                   "types_a": ",".join(sorted({r["type"] for r in ca})), "types_b": ",".join(sorted({r["type"] for r in cb}))}
            rga = [r for r in ca if r["type"] == "range"]
            rgb = [r for r in cb if r["type"] == "range"]
            if rga and rgb:
                row["range_jaccard"] = _interval_jaccard((max((r.get("low") or -np.inf) for r in rga), min((r.get("high") if r.get("high") is not None else np.inf) for r in rga)),
                                                         (max((r.get("low") or -np.inf) for r in rgb), min((r.get("high") if r.get("high") is not None else np.inf) for r in rgb)),
                                                         lo_ref, hi_ref)
            sa = [set(r["allowed"]) for r in ca if r["type"] == "category"]
            sb = [set(r["allowed"]) for r in cb if r["type"] == "category"]
            if sa and sb:
                A, Bs = set.intersection(*sa), set.intersection(*sb)
                row["category_jaccard"] = len(A & Bs) / len(A | Bs) if A | Bs else 1.0
            rows.append(row)
    return pd.DataFrame(rows)
