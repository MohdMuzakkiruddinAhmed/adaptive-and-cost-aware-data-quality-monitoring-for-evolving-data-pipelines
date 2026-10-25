"""Cross-column rules (v2.1): functional dependencies and conditional constraints.

The single-column rule families cannot see an error that is only wrong in combination with another field: a zip
code that belongs to a different city, a "no children" flag on a record that claims a child exemption. Two rule
types cover these (rules/static_rules.py):

  dependency   determinant column(s) -> dependent column, with the mapping LEARNED from the warm-up batches
               (majority dependent value per determinant value seen at least ``min_support`` times)
  conditional  if column A takes one of the given values, column B must take one of the allowed values / lie in
               [low, high]

Two sources:
  discover_rules        data-driven: every single-column determinant / every low-cardinality condition column whose
                        dependency holds on the warm-up for all but ``max_rate`` of the records (approximate FD and
                        constant-CFD discovery in the spirit of TANE and CFD miners, restricted to one determinant)
  validate_candidates   LLM proposals (rules/llm_crosscolumn.py): the model names the STRUCTURE ("zip determines
                        city", "if has_child is N then child_exemp is 0"); mappings are learned from the warm-up and
                        every candidate passes the same statistical test on the warm-up as the single-column rules
                        (rules/validator.py: at most max_warmup_failure_rate of the warm-up records may fail)

Ground truth is never used. The warm-up may contain erroneous records; a dependency that they break more often
than ``max_rate`` is not admitted.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from rules.static_rules import check

SEP = "\x1f"


def _key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return pd.Series([SEP.join(v) for v in zip(*[df[c].astype(str) for c in cols])], index=df.index)


def learn_mapping(warm: pd.DataFrame, determinant: List[str], dependent: str, min_support: int = 2
                  ) -> Tuple[Dict[str, str], float, float]:
    """(mapping, violation rate among covered records, share of records covered) of determinant -> dependent."""
    sub = warm[determinant + [dependent]].dropna()
    if not len(sub):
        return {}, 1.0, 0.0
    k = _key(sub, determinant)
    v = sub[dependent].astype(str)
    counts = pd.DataFrame({"k": k.to_numpy(), "v": v.to_numpy()}).value_counts().reset_index(name="n")
    support = counts.groupby("k")["n"].sum()
    top = counts.sort_values(["k", "n", "v"], ascending=[True, False, True]).drop_duplicates("k").set_index("k")
    keep = support[support >= min_support].index
    mapping = {str(a): str(top.at[a, "v"]) for a in keep}
    covered = k.isin(keep)
    if not covered.any():
        return mapping, 1.0, 0.0
    viol = (v[covered] != k[covered].map(mapping)).mean()
    return mapping, float(viol), float(covered.mean() * len(sub) / len(warm))


def dependency_rule(determinant: List[str], dependent: str, mapping: Dict[str, str], source: str) -> Dict[str, Any]:
    return {"type": "dependency", "columns": list(determinant), "column": dependent, "mapping": mapping,
            "key": f"dependency:{'+'.join(determinant)}>{dependent}", "source": source}


def conditional_rule(a: str, values: Iterable[str], b: str, allowed=None, low=None, high=None,
                     source: str = "learned") -> Dict[str, Any]:
    values = sorted(map(str, values))
    r = {"type": "conditional", "if_column": a, "if_values": values, "column": b, "source": source,
         "key": f"conditional:{a}={'|'.join(values)}>{b}"}
    if allowed is not None:
        r["allowed"] = sorted(map(str, allowed))
    else:
        r["low"], r["high"] = low, high
    return r


def _is_numeric(s: pd.Series) -> bool:
    x = pd.to_numeric(s.dropna(), errors="coerce")
    return len(x) > 0 and x.notna().mean() >= 0.99


def discover_rules(warm: pd.DataFrame, columns: List[str], max_rate: float, min_support: int = 2,
                   min_coverage: float = 0.2, max_condition_values: int = 20, min_condition_share: float = 0.01,
                   max_constant_share: float = 0.95) -> List[Dict[str, Any]]:
    """Dependencies and conditional constants that hold on the warm-up (see module docstring)."""
    n = len(warm)
    rules: List[Dict[str, Any]] = []
    nunique = {c: warm[c].nunique(dropna=True) for c in columns}
    for a in columns:
        if nunique[a] < 2 or nunique[a] > 0.9 * n:          # (almost) unique: nothing repeats to learn from
            continue
        for b in columns:
            if b == a or nunique[b] < 2:
                continue
            if nunique[b] > nunique[a] and nunique[a] > 0:
                continue                       # a determinant must not have fewer values than its dependent ...
            if nunique[a] <= max_condition_values:
                continue                       # ... and low-cardinality columns are handled as conditions below
            mapping, viol, cover = learn_mapping(warm, [a], b, min_support)
            # failure rate over ALL warm-up records, the same admission test as every single-column rule
            if mapping and viol * cover <= max_rate and cover >= min_coverage:
                rules.append(dependency_rule([a], b, mapping, "learned"))
    for a in columns:
        if not 2 <= nunique[a] <= max_condition_values:
            continue
        sa = warm[a].astype(str).where(warm[a].notna())
        for b in columns:
            if b == a or nunique[b] < 2:
                continue
            sb = warm[b]
            numeric = _is_numeric(sb)
            vb = pd.to_numeric(sb, errors="coerce") if numeric else sb.astype(str).where(sb.notna())
            overall = vb.value_counts(normalize=True, dropna=True)
            for val, share in sa.value_counts(normalize=True, dropna=True).items():
                if share < min_condition_share:
                    continue
                sel = vb[(sa == val) & vb.notna()]
                if not len(sel):
                    continue
                vc = sel.value_counts(normalize=True)
                const, frac = vc.index[0], float(vc.iloc[0])
                if frac < 1.0 - max_rate or overall.get(const, 0.0) > max_constant_share:
                    continue
                rules.append(conditional_rule(a, [val], b, low=float(const), high=float(const)) if numeric else
                             conditional_rule(a, [val], b, allowed=[const]))
    return rules


# ------------------------------------------------------------------------------------------- LLM candidates
def normalise_candidate(c: Dict[str, Any], columns: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """LLM cross-column proposal -> rule skeleton (mapping not yet learned). Returns (rule, error)."""
    if not isinstance(c, dict):
        return None, "not an object"
    t = c.get("rule_type")
    if t == "dependency":
        det = c.get("determinant")
        det = [det] if isinstance(det, str) else det
        dep = c.get("dependent")
        if not isinstance(det, list) or not det or not all(isinstance(x, str) for x in det):
            return None, "dependency without determinant columns"
        if any(x not in columns for x in det) or dep not in columns:
            return None, f"unknown column in {det} -> {dep!r}"
        if dep in det:
            return None, "dependent column is also a determinant"
        return {"type": "dependency", "columns": list(det), "column": dep}, None
    if t == "conditional":
        cond, then = c.get("if") or {}, c.get("then") or {}
        a, b = cond.get("column"), then.get("column")
        vals = cond.get("values")
        vals = [vals] if isinstance(vals, (str, int, float)) else vals
        if a not in columns or b not in columns:
            return None, f"unknown column {a!r} or {b!r}"
        if a == b:
            return None, "condition and consequence on the same column"
        if not isinstance(vals, list) or not vals:
            return None, "conditional without condition values"
        allowed = then.get("allowed_values")
        lo, hi = then.get("min"), then.get("max")
        if allowed is not None:
            if not isinstance(allowed, list) or not allowed:
                return None, "empty allowed_values"
            return conditional_rule(a, vals, b, allowed=allowed, source="llm_xcol"), None
        for v in (lo, hi):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)):
                return None, "bounds are not numbers"
        if lo is None and hi is None:
            return None, "conditional without allowed_values or bounds"
        if lo is not None and hi is not None and lo > hi:
            return None, "min > max"
        return conditional_rule(a, vals, b, low=lo, high=hi, source="llm_xcol"), None
    return None, f"unknown cross-column rule type {t!r}"


def _sig(r: Dict[str, Any]) -> str:
    keep = {k: r.get(k) for k in ("type", "columns", "column", "if_column", "if_values", "allowed", "low", "high")}
    return json.dumps(keep, sort_keys=True, default=str)


def validate_candidates(candidates: List[Dict[str, Any]], warm_batches: List[pd.DataFrame], columns: List[str],
                        max_rate: float, min_support: int = 2, prefix: str = "xcol") -> List[Dict[str, Any]]:
    """Admission of LLM cross-column proposals on the warm-up: syntax, mapping learned from the warm-up,
    warm-up failure rate <= max_rate, exact duplicates removed. Same statuses as rules/validator.py."""
    warm = pd.concat(warm_batches, ignore_index=True)
    order = sorted(range(len(candidates)),
                   key=lambda i: -(candidates[i].get("confidence") or 0) if isinstance(candidates[i], dict) else 0)
    out: List[Dict[str, Any]] = [None] * len(candidates)
    seen = set()
    for i in order:
        c = candidates[i]
        rec = {"index": i, "candidate": c, "rule": None, "status": None, "detail": None, "warmup_failure_rate": None,
               "rule_type": c.get("rule_type") if isinstance(c, dict) else None}
        out[i] = rec
        rule, err = normalise_candidate(c, columns)
        if err:
            rec.update(status="invalid", detail=err)
            continue
        if rule["type"] == "dependency":
            mapping, _, cover = learn_mapping(warm, rule["columns"], rule["column"], min_support)
            if not mapping:
                rec.update(status="rejected", detail="no determinant value repeats in the warm-up")
                continue
            rule = dependency_rule(rule["columns"], rule["column"], mapping, "llm_xcol")
        rate = float(check(rule, warm).fillna(False).mean())
        rec["warmup_failure_rate"] = rate
        if rate > max_rate:
            rec.update(rule=rule, status="rejected", detail=f"fails {rate:.2%} of warm-up records")
            continue
        s = _sig(rule)
        if s in seen:
            rec.update(rule=rule, status="redundant", detail="identical to an admitted rule")
            continue
        seen.add(s)
        rule["key"] = f"{prefix}{i}:{rule['key']}"
        rec.update(rule=rule, status="accepted")
    return out
