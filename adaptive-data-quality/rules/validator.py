"""Rule validation (project description, Section 9).

Every candidate rule (from the LLM) passes through:

  1. Syntax validation      known rule type, existing column, well-formed fields
                            (numbers for ranges, a non-empty list for categories,
                            a regular expression that compiles)            -> else "invalid"
  2. Execution feasibility  the rule runs on a real batch without error    -> else "invalid"
  3. Statistical validation on the warm-up batches T1..T5 the rule may flag at most
                            ``max_warmup_failure_rate`` of records. Too-strict range and
                            category rules are relaxed once to the warm-up data
                            (range widened to the 0.1%/99.9% quantiles, categories
                            extended with values seen in >= 0.1% of records) -> "modified"
                            if that passes, otherwise                        -> "rejected"
  4. Redundancy detection   a rule that is identical to, logically looser than (e.g. a
                            wider range), or whose warm-up failures are a subset of, a rule
                            already kept for the same column adds nothing      -> "redundant"

Rules that pass are "accepted" (or "modified"). Candidates are processed in order of
the LLM's confidence, so the more confident of two overlapping rules is kept.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rules.llm_generator import RULE_TYPES
from rules.static_rules import check

RECORDS = "_records"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def normalise(c: Dict[str, Any], cfg: Dict[str, Any], columns: List[str]) -> Tuple[Optional[Dict], Optional[str]]:
    """LLM rule -> executable rule. Returns (rule, error)."""
    if not isinstance(c, dict):
        return None, "not an object"
    t, col = c.get("rule_type"), c.get("column")
    if t not in RULE_TYPES:
        return None, f"unknown rule type {t!r}"
    monitored = cfg["numeric"] + cfg["categorical"]
    if t == "uniqueness":
        return {"type": "uniqueness", "columns": [x for x in columns if x != "record_id"]}, None
    if col not in monitored:
        return None, f"unknown column {col!r}"
    if t == "completeness":
        return {"type": "completeness", "column": col}, None
    if t == "type":
        if col not in cfg["numeric"]:
            return None, "type rule on a non-numerical column"
        return {"type": "type", "column": col, "dtype": "numeric"}, None
    if t == "range":
        if col not in cfg["numeric"]:
            return None, "range rule on a non-numerical column"
        lo, hi = c.get("min"), c.get("max")
        if lo is None and hi is None:
            return None, "range without min and max"
        if (lo is not None and not _is_num(lo)) or (hi is not None and not _is_num(hi)):
            return None, "range bounds are not numbers"
        if lo is not None and hi is not None and lo > hi:
            return None, "min > max"
        return {"type": "range", "column": col, "low": lo, "high": hi}, None
    if t == "category":
        vals = c.get("allowed_values")
        if not isinstance(vals, list) or not vals:
            return None, "category rule without allowed_values"
        return {"type": "category", "column": col, "allowed": sorted({str(v) for v in vals})}, None
    if t == "pattern":
        pat = c.get("pattern")
        if not isinstance(pat, str) or not pat:
            return None, "pattern rule without a pattern"
        try:
            re.compile(pat)
        except re.error as exc:
            return None, f"regular expression does not compile: {exc}"
        return {"type": "pattern", "column": col, "pattern": pat}, None
    return None, "unsupported"


def _relax(rule: Dict[str, Any], warm: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if rule["type"] == "range":
        x = pd.to_numeric(warm[rule["column"]], errors="coerce").dropna()
        q_lo, q_hi = float(x.quantile(0.001)), float(x.quantile(0.999))
        new = dict(rule)
        if new.get("low") is not None:
            new["low"] = min(new["low"], q_lo)
        if new.get("high") is not None:
            new["high"] = max(new["high"], q_hi)
        return new
    if rule["type"] == "category":
        share = warm[rule["column"]].dropna().astype(str).value_counts(normalize=True)
        return dict(rule, allowed=sorted(set(rule["allowed"]) | set(share[share >= 0.001].index)))
    return None


def _subsumes(kept: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """True if every record failing ``new`` necessarily fails ``kept`` (``new`` is looser)."""
    if kept["type"] != new["type"] or kept.get("column") != new.get("column"):
        return False
    if new["type"] == "range":
        lo_k = -np.inf if kept.get("low") is None else kept["low"]
        hi_k = np.inf if kept.get("high") is None else kept["high"]
        lo_n = -np.inf if new.get("low") is None else new["low"]
        hi_n = np.inf if new.get("high") is None else new["high"]
        return lo_n <= lo_k and hi_n >= hi_k
    if new["type"] == "category":
        return set(kept["allowed"]) <= set(new["allowed"])
    return new["type"] in ("completeness", "type", "uniqueness")


def logically_redundant(kept: Dict[str, Any], new: Dict[str, Any]) -> bool:
    """True only if ``new`` is implied by ``kept`` from their PREDICATES (manuscript Section 5.2):
    every value failing ``new`` provably fails ``kept``. Same-type implications: identical rules, a
    wider range, a superset category list, an identical pattern, repeated completeness/type/uniqueness.
    The only cross-type implication used: a pattern is implied by a category list whose every allowed
    value matches the pattern. A category list is NEVER removed because of a pattern (a well-formed
    invalid code passes the pattern but not the list), which is the failure of the empirical test that
    removed a valid US-state list as "redundant" to ^[A-Z]{2}$."""
    if kept.get("column", RECORDS) != new.get("column", RECORDS):
        return False
    if kept["type"] == new["type"]:
        if new["type"] == "pattern":
            return kept["pattern"] == new["pattern"]
        return _subsumes(kept, new)
    if new["type"] == "pattern" and kept["type"] == "category":
        try:
            return all(re.fullmatch(new["pattern"], str(v)) for v in kept["allowed"])
        except re.error:
            return False
    return False


def validate(candidates: List[Dict[str, Any]], cfg: Dict[str, Any], warm_batches: List[pd.DataFrame],
             max_rate: float, redundancy: str = "empirical") -> List[Dict[str, Any]]:
    """Admission checks of Section 9. ``redundancy``:
      "empirical"  proof-of-concept test: identical, logically looser, OR warm-up failures a subset of a
                   kept rule's failures (can remove a stronger cross-type rule, e.g. a state list)
      "logical"    refinement: only predicate-level implications (logically_redundant); provenance of
                   the removed rule and of the rule that covers it is recorded either way."""
    if redundancy not in ("empirical", "logical"):
        raise ValueError(f"unknown redundancy mode {redundancy!r}")
    warm = pd.concat(warm_batches, ignore_index=True)
    first = warm_batches[0]
    order = sorted(range(len(candidates)),
                   key=lambda i: -(candidates[i].get("confidence") or 0) if isinstance(candidates[i], dict) else 0)
    out: List[Dict[str, Any]] = [None] * len(candidates)
    kept: List[Tuple[Dict[str, Any], np.ndarray]] = []

    for i in order:
        c = candidates[i]
        rec = {"index": i, "candidate": c, "rule": None, "status": None, "stage": None, "detail": None,
               "warmup_failure_rate": None,
               "confidence": c.get("confidence") if isinstance(c, dict) else None,
               "rule_type": c.get("rule_type") if isinstance(c, dict) else None,
               "column": c.get("column") if isinstance(c, dict) else None}
        out[i] = rec
        rule, err = normalise(c, cfg, list(first.columns))
        if err:
            rec.update(status="invalid", stage="syntax", detail=err); continue
        try:
            check(rule, first)
        except Exception as exc:  # noqa: BLE001
            rec.update(status="invalid", stage="execution", detail=f"{type(exc).__name__}: {exc}"); continue

        mask = check(rule, warm).fillna(False).to_numpy(dtype=bool)
        rate = float(mask.mean())
        rec["warmup_failure_rate"] = rate
        status = "accepted"
        if rate > max_rate:
            relaxed = _relax(rule, warm)
            if relaxed is None:
                rec.update(rule=rule, status="rejected", stage="statistical",
                           detail=f"fails {rate:.2%} of warm-up records"); continue
            mask = check(relaxed, warm).fillna(False).to_numpy(dtype=bool)
            if mask.mean() > max_rate:
                rec.update(rule=relaxed, status="rejected", stage="statistical",
                           detail=f"fails {rate:.2%} of warm-up records, {mask.mean():.2%} after relaxing"); continue
            rule, status = relaxed, "modified"
            rec.update(detail=f"relaxed: fails {rate:.2%} before, {mask.mean():.2%} after")
            rec["warmup_failure_rate"] = float(mask.mean())

        col = rule.get("column", RECORDS)
        redundant_to, redundant_idx = None, None
        for k_idx, k_rule, k_mask in kept:
            if k_rule.get("column", RECORDS) != col:
                continue
            if redundancy == "logical":
                hit = logically_redundant(k_rule, rule)
            else:
                same = json.dumps(k_rule, sort_keys=True) == json.dumps(rule, sort_keys=True)
                subset = mask.any() and not (mask & ~k_mask).any()
                hit = same or subset or _subsumes(k_rule, rule)
            if hit:
                redundant_to, redundant_idx = k_rule, k_idx
                break
        rec["redundancy_mode"] = redundancy
        if redundant_to is not None:
            rec.update(rule=rule, status="redundant", stage="redundancy", redundant_to_index=redundant_idx,
                       detail=f"covered by candidate {redundant_idx}: {json.dumps(redundant_to)[:120]}"); continue
        rec.update(rule=rule, status=status, stage=rec["stage"] or "passed")
        kept.append((i, rule, mask))
    return out


def accepted_rules(records: List[Dict[str, Any]], prefix: str = "llm") -> List[Dict[str, Any]]:
    """Executable rules that passed validation, each with a unique check key."""
    rules = []
    for r in records:
        if r["status"] in ("accepted", "modified"):
            rule = dict(r["rule"])
            rule["key"] = f"{prefix}{r['index']}:{rule['type']}:{rule.get('column') or RECORDS}"
            rules.append(rule)
    return rules


def executable_rules(records: List[Dict[str, Any]], candidates: List[Dict[str, Any]], cfg: Dict[str, Any],
                     columns: List[str], prefix: str = "llmraw") -> List[Dict[str, Any]]:
    """Every rule that passed syntax and execution checks, as the LLM wrote it (no statistical
    validation, no redundancy removal): the 'LLM rules' condition of Experiment 3."""
    rules = []
    for r in records:
        if r["status"] == "invalid":
            continue
        rule, _ = normalise(candidates[r["index"]], cfg, columns)
        rule["key"] = f"{prefix}{r['index']}:{rule['type']}:{rule.get('column') or RECORDS}"
        rules.append(rule)
    return rules
