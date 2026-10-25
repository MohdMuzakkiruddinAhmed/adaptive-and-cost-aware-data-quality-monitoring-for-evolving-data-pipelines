"""Check schedulers for RQ3 / H3 (supervisor experiment 5; manuscript Sections 5.6 and 6.4).

All schedulers see the same candidate suite and the same mandatory checks: "basic" tier checks
(completeness, type, range; experiment.yaml risk.tiers) run on every batch. They differ only in which
"standard" checks (category, uniqueness, pattern) run:

  all           every check, every batch (run-all baseline)
  periodic      check j runs at evenly spaced batches with rate ``rate`` (default 1 / period): it runs at t when
                floor(t * rate + u_j) > floor((t - 1) * rate + u_j), u_j = 0 ("aligned") or a stable hash of the
                check key in [0, 1) ("staggered", spreads cost evenly)
  random        each standard check runs with probability ``rate`` (seeded, reproducible)
  failure_only  risk policy with weights F = 1, all others 0
  drift_only    risk policy with weights D = 1, all others 0
  risk          proposed: R = 0.25F + 0.25D + 0.15S + 0.20H + 0.15I with Section 12 thresholds,
                refresh every ``refresh_every`` batches, complete suite when any unit is "very high"

Cost matching (protocol.yaml schedulers.cost_match): the period and the random rate can be set from the
risk policy's executed share of standard checks in the same run, so all schedulers are compared at
(approximately) the same number of executed checks.

Stale observations (Section 5.6): a skipped check supplies no new failure observation. The risk tracker
keeps its last observed rate; every scheduler reports, per batch, the share of checks executed and the
age (in batches) of the oldest observation of each standard check.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from optimizer.rule_selector import select as risk_select
from risk.risk_score import RECORDS, RiskTracker, calibrate_f, check_key

POLICIES = ["all", "periodic", "random", "failure_only", "drift_only", "risk"]
RISK_LIKE = {"risk": None, "failure_only": {"F": 1.0, "D": 0.0, "S": 0.0, "H": 0.0, "I": 0.0},
             "drift_only": {"F": 0.0, "D": 1.0, "S": 0.0, "H": 0.0, "I": 0.0}}


def _stable_offset(key: str, period: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % max(period, 1)


class Scheduler:
    def __init__(self, policy: str, exp: Dict[str, Any], units: List[str], importance: Dict[str, float],
                 period: Optional[int] = None, rate: Optional[float] = None, phase: str = "staggered",
                 seed: int = 0, weights: Optional[Dict[str, float]] = None,
                 thresholds: Optional[Dict[str, float]] = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown scheduler {policy!r}")
        r = exp["risk"]
        self.policy, self.tiers, self.refresh = policy, r["tiers"], r["refresh_every"]
        self.units, self.importance = units, importance
        self.weights = weights or (RISK_LIKE.get(policy) or r["weights"])
        self.thr = dict(thresholds or r["thresholds"])
        self.window = r["history_window"]
        self.period = int(period or self.refresh)
        self.rate = 1.0 / self.period if rate is None else float(rate)
        self.phase = phase
        self.rng = np.random.default_rng([seed, 7919])
        self.tracker: Optional[RiskTracker] = None
        self.last_run: Dict[str, int] = {}
        self.last_risk: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def uses_risk(self) -> bool:
        return self.policy in RISK_LIKE

    def calibrate(self, warm_rates: List[Dict[str, float]], n_records: int, drift_thresholds: Dict[str, float]) -> None:
        """Called once at the end of the warm-up with the per-check failure rates of T2..T5."""
        if self.uses_risk:
            self.tracker = RiskTracker(self.units, self.importance, self.weights, calibrate_f(warm_rates, n_records),
                                       drift_thresholds, self.window)

    def select(self, rules: List[Dict[str, Any]], t: int, d_raw: Dict[str, float],
               schema_changed: Dict[str, bool]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {"policy": self.policy}
        if self.policy == "all":
            chosen = list(rules)
        elif self.uses_risk:
            risk = self.tracker.score(d_raw, schema_changed)
            self.last_risk = risk
            chosen, sel = risk_select(rules, risk, self.tiers, self.thr, self.last_run, t, self.refresh)
            info.update(sel)
            info["max_risk"] = max(v["R"] for v in risk.values())
            info["risk"] = {u: v["R"] for u, v in risk.items()}
        else:
            chosen = []
            for j, r in enumerate(rules):
                key = check_key(r)
                if self.tiers[r["type"]] == "basic":
                    chosen.append(r)
                elif self.policy == "periodic":
                    # deterministic, evenly spaced execution at rate self.rate (exactly cost-matched, also for
                    # rates above 0.5 where an integer period would round to "every batch"); staggered checks
                    # get different phases u in [0, 1)
                    u = (_stable_offset(key, 10 ** 6) / 10 ** 6) if self.phase == "staggered" else 0.0
                    if np.floor(t * self.rate + u) > np.floor((t - 1) * self.rate + u):
                        chosen.append(r)
                elif self.policy == "random":
                    if self.rng.random() < self.rate:
                        chosen.append(r)
        std = [r for r in rules if self.tiers[r["type"]] != "basic"]
        chosen_keys = {check_key(r) for r in chosen}
        info["standard_total"] = len(std)
        info["standard_executed"] = sum(1 for r in std if check_key(r) in chosen_keys)
        info["coverage"] = len(chosen) / len(rules) if rules else 1.0
        ages = [t - self.last_run.get(check_key(r), 0) for r in std if check_key(r) not in chosen_keys]
        info["max_staleness"] = int(max(ages)) if ages else 0
        return chosen, info

    def update(self, t: int, observed: Dict[str, float]) -> None:
        for k in observed:
            self.last_run[k] = t
        if self.tracker is not None and self.last_risk is not None:
            self.tracker.update(observed, self.last_risk)

    def mark_warmup(self, t: int, observed: Dict[str, float]) -> None:
        for k in observed:
            self.last_run[k] = t
