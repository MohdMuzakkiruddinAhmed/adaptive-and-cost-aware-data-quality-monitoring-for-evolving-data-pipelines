"""Manually authored domain rules (configs/manual_rules.yaml): the manual-rule condition of RQ2 and the
manual constraints of the H2b category guard. Manual rules use the LLM rule format, so they are turned
into executable rules by the same code (rules.validator.normalise)."""
from __future__ import annotations

from typing import Any, Dict, List

from common.utils import load_yaml
from rules.validator import RECORDS, normalise


def manual_config() -> Dict[str, Any]:
    return load_yaml("manual_rules")


def manual_candidates(dataset: str, manual: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    manual = manual or manual_config()
    return [dict(r, confidence=1.0) for r in (manual.get("rules", {}).get(dataset) or [])]


def manual_rules(dataset: str, cfg: Dict[str, Any], columns: List[str],
                 manual: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Executable manual rules, each with a unique check key ("manual<i>:<type>:<column>")."""
    out = []
    for i, c in enumerate(manual_candidates(dataset, manual)):
        rule, err = normalise(c, cfg, columns)
        if err:
            raise ValueError(f"manual rule {i} for {dataset} is invalid: {err} ({c})")
        rule["key"] = f"manual{i}:{rule['type']}:{rule.get('column') or RECORDS}"
        rule["source"] = "manual"
        out.append(rule)
    return out


def validate_manual(manual: Dict[str, Any] | None = None) -> bool:
    """configs/manual_rules.yaml ``validate_manual`` (v2.2): admit manual rules with the LLM admission test."""
    return bool((manual or manual_config()).get("validate_manual", False))


def admitted_manual_rules(dataset: str, cfg: Dict[str, Any], warm_batches, max_rate: float,
                          manual: Dict[str, Any] | None = None, redundancy: str = "logical"):
    """(executable manual rules, admission records) after the SAME admission test as the LLM rules
    (rules/validator.py validate) on this run's own warm-up batches T1..T5 (training data only).

    v2.2: an expert rule is tested against history before it is deployed, as the LLM rules are; without it a
    dictionary-only author's factual slip goes straight into production (dev run 4: the BTS carrier pattern
    ^[A-Z]{2}$ rejects B6 / F9 / 9E / G4 = 11% of valid records; the Chicago Results list omits Out of Business,
    No Entry and Not Ready = 13%). Logical redundancy mode, as in the category guard (keeps cross-type rules)."""
    from rules.validator import accepted_rules, validate
    cands = manual_candidates(dataset, manual)
    val = validate(cands, cfg, warm_batches, max_rate, redundancy)
    rules = []
    for r in accepted_rules(val, prefix="manual"):
        r["source"] = "manual"
        rules.append(r)
    return rules, val, cands


def domain_constraints(rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Category and pattern rules per column: the domain constraints used by the category guard."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rules:
        if r["type"] in ("category", "pattern"):
            out.setdefault(r["column"], []).append(r)
    return out
