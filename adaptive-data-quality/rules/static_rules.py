"""Static rule engine (project description, Section 8.1).

Rule types: completeness, range, uniqueness, category, type. A rule is a dict; ``check``
returns a boolean Series that is True where a record FAILS the rule.

    {"type": "completeness", "column": "amount"}                  amount IS NOT NULL
    {"type": "range", "column": "age", "low": 0, "high": 120}     0 <= age <= 120
    {"type": "uniqueness", "columns": [...]}                      no repeated record
    {"type": "category", "column": "status", "allowed": [...]}    status in {...}
    {"type": "type", "column": "amount", "dtype": "numeric"}      amount must be numeric
    {"type": "pattern", "column": "code", "pattern": "^[A-Z]{3}$"} value matches a regular expression
                                                                  (used by LLM-generated rules, Section 8.3)
Cross-column rules (rules/crosscolumn.py; external benchmarks, v2.1):
    {"type": "dependency", "columns": ["zip"], "column": "city", "mapping": {"01907": "SWAMPSCOTT", ...}}
                                     the determinant value(s) fix the dependent value: a record whose determinant
                                     is in the mapping but whose dependent value differs fails (functional
                                     dependency learned from the warm-up; missing values never fail)
    {"type": "conditional", "if_column": "has_child", "if_values": ["N"], "column": "child_exemp",
     "low": 0, "high": 0}  (or "allowed": [...])
                                     when the condition holds, the value must lie in [low, high] (numeric) or in
                                     allowed (text); a missing or non-numeric value never fails (other checks
                                     cover those)
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def check(rule: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    t = rule["type"]
    if t == "completeness":
        return df[rule["column"]].isna()
    if t == "range":
        x = pd.to_numeric(df[rule["column"]], errors="coerce")
        low = rule.get("low") if rule.get("low") is not None else -float("inf")
        high = rule.get("high") if rule.get("high") is not None else float("inf")
        return x.notna() & ((x < low) | (x > high))
    if t == "uniqueness":
        return df.duplicated(subset=rule["columns"], keep="first")
    if t == "category":
        s = df[rule["column"]]
        return s.notna() & ~s.astype(str).isin(set(rule["allowed"]))
    if t == "type":
        s = df[rule["column"]]
        return s.notna() & pd.to_numeric(s, errors="coerce").isna()
    if t == "pattern":
        s = df[rule["column"]]
        return s.notna() & ~s.astype(str).str.fullmatch(rule["pattern"])
    if t == "dependency":
        det = [df[c] for c in rule["columns"]]
        dep = df[rule["column"]]
        ok = dep.notna()
        for d in det:
            ok &= d.notna()
        key = det[0].astype(str) if len(det) == 1 else \
            pd.Series(["\x1f".join(v) for v in zip(*[d.astype(str) for d in det])], index=df.index)
        expected = key.map(rule["mapping"])
        return ok & expected.notna() & (dep.astype(str) != expected)
    if t == "conditional":
        cond = df[rule["if_column"]].astype(str).isin(set(map(str, rule["if_values"]))) & df[rule["if_column"]].notna()
        s = df[rule["column"]]
        if rule.get("allowed") is not None:
            bad = s.notna() & ~s.astype(str).isin(set(map(str, rule["allowed"])))
        else:
            x = pd.to_numeric(s, errors="coerce")
            low = rule.get("low") if rule.get("low") is not None else -float("inf")
            high = rule.get("high") if rule.get("high") is not None else float("inf")
            bad = x.notna() & ((x < low) | (x > high))
        return cond & bad
    raise ValueError(f"unknown rule type {t}")


def involved_columns(rule: Dict[str, Any]) -> List[str]:
    """Columns a rule reads (all cells of a violating tuple are reported for cross-column rules)."""
    t = rule["type"]
    if t == "uniqueness":
        return list(rule["columns"])
    if t == "dependency":
        return list(rule["columns"]) + [rule["column"]]
    if t == "conditional":
        return [rule["if_column"], rule["column"]]
    return [rule["column"]]


def describe(rule: Dict[str, Any]) -> str:
    t = rule["type"]
    if t == "completeness":
        return f"{rule['column']} IS NOT NULL"
    if t == "range":
        lo = "-inf" if rule.get("low") is None else f"{rule['low']:.4g}"
        hi = "inf" if rule.get("high") is None else f"{rule['high']:.4g}"
        return f"{lo} <= {rule['column']} <= {hi}"
    if t == "pattern":
        return f"{rule['column']} matches {rule['pattern']}"
    if t == "uniqueness":
        return "record is unique"
    if t == "category":
        return f"{rule['column']} in {{{len(rule['allowed'])} categories}}"
    if t == "type":
        return f"{rule['column']} is numeric"
    if t == "dependency":
        return f"{' + '.join(rule['columns'])} determines {rule['column']} ({len(rule['mapping'])} mapped values)"
    if t == "conditional":
        then = f"in {{{len(rule['allowed'])} values}}" if rule.get("allowed") is not None else \
            f"in [{rule.get('low')}, {rule.get('high')}]"
        return f"if {rule['if_column']} in {rule['if_values']} then {rule['column']} {then}"
    return str(rule)


def base_rules(df_columns: List[str], numeric: List[str], categorical: List[str],
               nullable: List[str] = ()) -> List[Dict[str, Any]]:
    """Rules that need no statistics: completeness, uniqueness and type. Columns listed in ``nullable``
    (legitimately empty values, e.g. in external benchmarks) get no completeness rule."""
    rules: List[Dict[str, Any]] = [{"type": "completeness", "column": c} for c in numeric + categorical
                                   if c not in set(nullable)]
    rules.append({"type": "uniqueness", "columns": [c for c in df_columns if c != "record_id"]})
    rules += [{"type": "type", "column": c, "dtype": "numeric"} for c in numeric]
    return rules
