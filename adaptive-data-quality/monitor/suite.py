"""Candidate rule suite Q_t of one monitoring configuration (manuscript Section 5.1).

Q_t = base checks (completeness, type, whole-record uniqueness)
    + statistical checks of the chosen threshold method (adaptation/baselines.py)
    + category checks of the chosen admission policy (adaptation/category_guard.py), if one is set
    + extra rules (validated LLM rules, raw LLM rules, manual domain rules)

Every rule carries a unique ``key`` and a ``source`` (base | stat | llm | llm_raw | manual).

Exact duplicates of base checks among the extra rules (e.g. an LLM "completeness" rule on a column the
base suite already checks) are dropped from execution but kept in ``dropped`` with their provenance
(manuscript Section 5.2: "removes exact duplicates without discarding provenance"); set
``dedupe_extra=False`` to execute them anyway, as the proof of concept did.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from adaptation.baselines import category_rules, numeric_rules
from adaptation.category_guard import CategoryGuard
from rules.static_rules import base_rules
from rules.validator import logically_redundant


def _with_key(rule: Dict[str, Any], source: str) -> Dict[str, Any]:
    r = dict(rule)
    r.setdefault("key", f"{r['type']}:{r.get('column') or '_records'}")
    r["source"] = r.get("source") or source
    return r


class SuiteBuilder:
    def __init__(self, cfg: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any], method: str,
                 category_policy: Optional[str] = None, guard_constraints: Optional[Dict[str, List]] = None,
                 extra_rules: Optional[List[Dict[str, Any]]] = None, extra_source: str = "llm",
                 dedupe_extra: bool = True, include_statistical: bool = True, include_base: bool = True) -> None:
        self.cfg, self.exp, self.protocol = cfg, exp, protocol
        methods = protocol["baselines"]["methods"]
        if method not in methods:
            raise ValueError(f"unknown method {method!r}; defined: {list(methods)}")
        self.method_name, self.method = method, methods[method]
        self.params = {k: protocol["baselines"][k] for k in ("ewma_lambda", "iqr_k", "mad_k")}
        self.keep_warmup = bool(exp["adaptive"].get("category_keep_warmup", False))
        self.guard = CategoryGuard(category_policy, cfg["categorical"], exp["adaptive"]["category_min_batches"],
                                   guard_constraints, self.keep_warmup) if category_policy else None
        self.extra = [_with_key(r, r.get("source") or extra_source) for r in (extra_rules or [])]
        self.dedupe_extra = dedupe_extra
        self.include_statistical = include_statistical
        self.include_base = include_base        # False with include_statistical=False: "LLM rules alone"
        self.dropped: List[Dict[str, Any]] = []

    def build(self, columns: List[str], warm: List[Dict[str, Any]], history: List[Dict[str, Any]],
              past: List[Dict[str, Any]], current: Optional[Dict[str, Any]]
              ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
        num, cat = self.cfg["numeric"], self.cfg["categorical"]
        rules = [_with_key(r, "base") for r in base_rules(columns, num, cat, self.cfg.get("nullable", []))] \
            if self.include_base else []
        admissions: Dict[str, Dict[str, str]] = {}
        # columns with a physical domain (datasets.yaml domain_ranges, e.g. coordinates inside the city) get the
        # domain range as a base check in EVERY method instead of a statistical k-sigma range
        domain = self.cfg.get("domain_ranges") or {}
        if self.include_base:
            rules += [_with_key({"type": "range", "column": c, "low": float(lo), "high": float(hi), "policy": "domain"},
                                "base") for c, (lo, hi) in domain.items() if c in columns]
        if self.include_statistical:
            stat_num = [c for c in num if c not in domain]
            rules += [_with_key(r, "stat") for r in numeric_rules(self.method["numeric"], stat_num,
                                                                  self.exp["range_k"], warm, history, past, current,
                                                                  self.params)]
            if self.guard is not None:
                cats, admissions = self.guard.rules(warm, history)
            else:
                cats = category_rules(self.method["category"], cat, warm, history,
                                      self.exp["adaptive"]["category_min_batches"], self.keep_warmup)
            rules += [_with_key(r, "stat") for r in cats]
        base = [r for r in rules if r["source"] == "base"]
        self.dropped = []
        for r in self.extra:
            if self.dedupe_extra and r["type"] in ("completeness", "type", "uniqueness") and \
                    any(logically_redundant(b, r) for b in base):
                self.dropped.append({"key": r["key"], "reason": "duplicate of a base check"})
                continue
            rules.append(r)
        return rules, admissions
