"""Statistical rule generation from profiles (project description, Section 8.2).

Range rules start from mu +/- k*sigma (k = 3); category rules from the observed categories.
"""
from __future__ import annotations

from typing import Any, Dict, List


def range_rules(profile: Dict[str, Any], numeric: List[str], k: float) -> List[Dict[str, Any]]:
    out = []
    for c in numeric:
        p = profile["columns"][c]
        out.append({"type": "range", "column": c, "low": p["mean"] - k * p["std"], "high": p["mean"] + k * p["std"]})
    return out


def category_rules(profile: Dict[str, Any], categorical: List[str]) -> List[Dict[str, Any]]:
    return [{"type": "category", "column": c, "allowed": sorted(profile["columns"][c]["frequencies"])}
            for c in categorical]
