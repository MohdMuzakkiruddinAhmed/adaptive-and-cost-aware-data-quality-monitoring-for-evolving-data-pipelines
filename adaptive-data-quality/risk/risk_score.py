"""Risk score (project description, Section 11):

    R_i = w1*F_i + w2*D_i + w3*S_i + w4*H_i + w5*I_i        (w = 0.25, 0.25, 0.15, 0.20, 0.15)

Units i are the experiment columns plus "_records" (the record-level unit checked by the
uniqueness rule). Every component is normalised to [0, 1]:

F_i  recent failure frequency. For each check of the unit, its most recent observed failure
     rate f is compared with its warm-up normal level (mu, sd):
         F_check = clip((f - mu) / (3 * sd), 0, 1)
     so F = 1 means "failing clearly above normal". F_i is the largest F_check of the unit.
     A check that was skipped keeps its last observed rate.
D_i  distribution drift of the column, current profile vs. the warm-up profiles, divided by
     the column's no-drift threshold (mean + 3 sd over the warm-up): clip(D / tau, 0, 1).
     0 for "_records".
S_i  schema-change risk: 1 if the column is missing or its data type changed since the
     previous batch, else 0. For "_records": 1 if any column changed.
H_i  historical instability: share of the last k batches in which the unit was anomalous
     (F_i = 1 or D_i = 1).
I_i  downstream importance from configs/datasets.yaml.

Risk is computed before any check runs on the batch. It uses only the batch profile
(always computed, Section 5) and the results of earlier batches.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

import numpy as np

RECORDS = "_records"


def check_key(rule: Dict[str, Any]) -> str:
    """Identifier of a check. LLM rules carry their own key (e.g. "llm3:range:DepDelay")."""
    return rule.get("key") or f"{rule['type']}:{rule.get('column') or RECORDS}"


def unit_of(key: str) -> str:
    return key.rsplit(":", 1)[1]


class RiskTracker:
    def __init__(self, units: List[str], importance: Dict[str, float], weights: Dict[str, float],
                 f_baseline: Dict[str, Dict[str, float]], d_threshold: Dict[str, float], window: int) -> None:
        self.units = units
        self.importance = importance
        self.w = weights
        self.f_base = f_baseline               # check key -> {"mu": .., "sd": ..}
        self.tau = d_threshold                 # column -> no-drift threshold
        self.last_f: Dict[str, float] = {k: v["mu"] for k, v in f_baseline.items()}
        self.anomalous = {u: deque(maxlen=window) for u in units}

    def _f_check(self, key: str) -> float:
        b = self.f_base.get(key)
        if b is None:
            return 0.0
        f = self.last_f.get(key, b["mu"])
        if b["sd"] <= 0:
            return 1.0 if f > b["mu"] else 0.0
        return float(np.clip((f - b["mu"]) / (3 * b["sd"]), 0, 1))

    def score(self, d_raw: Dict[str, float], schema_changed: Dict[str, bool]) -> Dict[str, Dict[str, float]]:
        out = {}
        any_change = any(schema_changed.values())
        for u in self.units:
            f = max([self._f_check(k) for k in self.f_base if unit_of(k) == u] or [0.0])
            v, tau = d_raw.get(u), self.tau.get(u)
            d = float(np.clip(v / tau, 0, 1)) if (v is not None and tau is not None and np.isfinite(tau) and tau > 0) else 0.0
            s = 1.0 if (any_change if u == RECORDS else schema_changed.get(u, False)) else 0.0
            h = float(np.mean(self.anomalous[u])) if self.anomalous[u] else 0.0
            i = float(self.importance.get(u, 0.5))
            r = self.w["F"] * f + self.w["D"] * d + self.w["S"] * s + self.w["H"] * h + self.w["I"] * i
            out[u] = {"F": f, "D": d, "S": s, "H": h, "I": i, "R": r}
        return out

    def update(self, observed: Dict[str, float], scores: Dict[str, Dict[str, float]]) -> None:
        """Record failure rates of the checks that ran and whether each unit was anomalous."""
        self.last_f.update(observed)
        for u in self.units:
            f_now = max([self._f_check(k) for k in self.f_base if unit_of(k) == u] or [0.0])
            self.anomalous[u].append(1.0 if (f_now >= 1.0 or scores[u]["D"] >= 1.0) else 0.0)


def calibrate_f(warmup_rates: List[Dict[str, float]], n_records: int) -> Dict[str, Dict[str, float]]:
    """Warm-up mean and sd of each check's failure rate. The sd is floored at the binomial
    sampling sd so that a check that almost never fails is not infinitely sensitive."""
    keys = sorted(set().union(*warmup_rates))
    out = {}
    for k in keys:
        x = np.array([r.get(k, 0.0) for r in warmup_rates])
        mu = float(x.mean())
        p = min(max(mu, 1.0 / n_records), 1.0 - 1.0 / n_records)   # also a check failing on (almost) every record
        sd = max(float(x.std(ddof=1)) if len(x) > 1 else 0.0, float(np.sqrt(p * (1 - p) / n_records)))
        out[k] = {"mu": mu, "sd": sd}
    return out
