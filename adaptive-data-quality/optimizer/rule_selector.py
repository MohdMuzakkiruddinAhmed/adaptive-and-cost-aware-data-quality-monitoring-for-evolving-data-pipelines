"""Cost-aware rule selector (project description, Section 12).

Checks are tiered by relative cost, following Section 11 (missingness and range = low,
uniqueness = medium):
    basic     completeness, type, range     -> always run
    standard  category, uniqueness          -> run when the unit's risk calls for it

Risk levels (Section 12 initial thresholds):
    Low        R < 0.20          basic checks only
    Moderate   0.20 <= R < 0.50  basic + the standard checks of the selected (moderate-or-higher) units
    High       0.50 <= R < 0.75  same at unit level (all standard checks of the unit)
    Very High  R >= 0.75         complete suite for the whole batch

Refresh: a check that has not run for ``refresh_every`` batches runs anyway, so its failure
history (F in the risk score) never goes stale indefinitely.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from risk.risk_score import check_key, unit_of


def level(r: float, thr: Dict[str, float]) -> str:
    if r >= thr["very_high"]:
        return "very_high"
    if r >= thr["high"]:
        return "high"
    if r >= thr["moderate"]:
        return "moderate"
    return "low"


def select(rules: List[Dict[str, Any]], risk: Dict[str, Dict[str, float]], tiers: Dict[str, str],
           thr: Dict[str, float], last_run: Dict[str, int], t: int, refresh_every: int) -> Tuple[List[Dict], Dict]:
    levels = {u: level(v["R"], thr) for u, v in risk.items()}
    full = any(lv == "very_high" for lv in levels.values())
    chosen, reasons = [], {"basic": 0, "risk": 0, "refresh": 0, "very_high": 0, "skipped": 0}
    for r in rules:
        key = check_key(r)
        if tiers[r["type"]] == "basic":
            chosen.append(r); reasons["basic"] += 1
        elif full:
            chosen.append(r); reasons["very_high"] += 1
        elif levels.get(unit_of(key), "low") != "low":
            chosen.append(r); reasons["risk"] += 1
        elif t - last_run.get(key, -10**9) >= refresh_every:
            chosen.append(r); reasons["refresh"] += 1
        else:
            reasons["skipped"] += 1
    return chosen, {"levels": levels, "complete_suite": full, **reasons}
