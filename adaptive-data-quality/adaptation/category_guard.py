"""Category-admission policies for H2b (manuscript Section 5.4, Figure 2; supervisor experiment 4).

A statistical category rule decides which values of a categorical column are accepted. Learning new
categories from recurrence alone reduces false alarms on legitimate novelty, but it also learns a
recurring INVALID value after two batches. The guard keeps the recurrence requirement and adds a check
against applicable, independently justified domain constraints:

  frozen             the warm-up categories, never extended (controls for "never accept novelty")
  recurrence         proof-of-concept rule: seen in >= min_batches of the previous window batches
  recurrence_manual  recurrence AND compatible with the manual domain constraints of the column
  recurrence_llm     recurrence AND compatible with the validated LLM constraints of the column
  recurrence_llm_value  recurrence AND the LLM, asked about the value itself, did not call it an error
                     (experiments/v2/llm_value_checks.py; constraints = {column: {value: verdict}})

Admission status of each recurring value (recorded for every batch, so acceptance times can be measured):

  admitted_by_list       in every closed category list of the column (domain constraint says valid)
  rejected               violates a category list or a pattern: never admitted, however frequent
  admitted_compatible    passes the column's patterns, and the column has no closed list: compatible with
                         the constraints, NOT shown to be valid (open domain, reported as unresolved)
  admitted_unconstrained no applicable constraint for the column (unresolved)
  admitted_recurrence    recurrence-only policy (no constraint consulted)
  admitted_llm_valid     the LLM judged the value legitimate (recurrence_llm_value)
  admitted_llm_unsure    the LLM was unsure or was not asked: recurrence decides (unresolved)

Frequency alone never overrides a violated constraint (Section 5.4).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from adaptation.adaptive_thresholds import adaptive_category_rules

POLICIES = ["frozen", "recurrence", "recurrence_manual", "recurrence_llm", "recurrence_llm_value"]
UNRESOLVED = ("admitted_compatible", "admitted_unconstrained", "admitted_llm_unsure")
VERDICT_STATUS = {"valid": "admitted_llm_valid", "invalid": "rejected", "unsure": "admitted_llm_unsure"}


def admission_status(value: str, constraints: List[Dict[str, Any]]) -> str:
    lists = [c for c in constraints if c["type"] == "category"]
    patterns = [c for c in constraints if c["type"] == "pattern"]
    if any(value not in set(map(str, c["allowed"])) for c in lists):
        return "rejected"
    for p in patterns:
        try:
            if not re.fullmatch(p["pattern"], value):
                return "rejected"
        except re.error:
            continue
    if lists:
        return "admitted_by_list"
    if patterns:
        return "admitted_compatible"
    return "admitted_unconstrained"


class CategoryGuard:
    """Category rules under one admission policy.

    ``constraints``: column -> list of executable category/pattern rules (manual or validated LLM rules);
    required for the guarded policies, ignored otherwise."""

    def __init__(self, policy: str, categorical: List[str], min_batches: int,
                 constraints: Optional[Dict[str, List[Dict[str, Any]]]] = None, keep_warmup: bool = False) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown category policy {policy!r}")
        self.policy, self.categorical, self.min_batches = policy, categorical, min_batches
        self.constraints = constraints or {}
        self.keep_warmup = keep_warmup

    def rules(self, warm: List[Dict[str, Any]], history: List[Dict[str, Any]]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
        """(category rules for the current batch, {column: {value: admission status}})."""
        if self.policy == "frozen":
            out = []
            for c in self.categorical:
                allowed = sorted(set().union(*[set(p["columns"][c]["frequencies"]) for p in warm]))
                out.append({"type": "category", "column": c, "allowed": allowed, "policy": "frozen"})
            return out, {c: {} for c in self.categorical}
        base = adaptive_category_rules(history, self.categorical, self.min_batches,
                                       warm if self.keep_warmup else None)
        out, status = [], {}
        for r in base:
            c = r["column"]
            if self.policy == "recurrence":
                st = {v: "admitted_recurrence" for v in r["allowed"]}
            elif self.policy == "recurrence_llm_value":
                warm_vals = set().union(*[set(p["columns"][c]["frequencies"]) for p in warm]) if warm else set()
                verdicts = self.constraints.get(c, {})
                st = {v: ("admitted_recurrence" if str(v) in warm_vals else
                          VERDICT_STATUS.get(verdicts.get(str(v), "unsure"), "admitted_llm_unsure"))
                      for v in r["allowed"]}
            else:
                cons = self.constraints.get(c, [])
                st = {v: admission_status(str(v), cons) for v in r["allowed"]}
            status[c] = st
            out.append({"type": "category", "column": c, "policy": self.policy,
                        "allowed": sorted(v for v, s in st.items() if s != "rejected")})
        return out, status
